import os
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
import json
from langchain.schema import SystemMessage
import chainlit as cl
from newsplease import NewsPlease
import time
from duckduckgo_search import DDGS
from itertools import islice


def search(query, max_retries=5):
    """
    Search the given query using DuckDuckGo.
    Args:
    - query (str): The search query.
    - max_retries (int): Maximum number of retries in case of request failure.
    Returns:
    - list[dict]: A list of search results with 'title' and 'url'.
    """
    for attempt in range(max_retries):
        try:
            result = []

            # Initialize the DuckDuckGo search object.
            with DDGS() as ddgs:
                response = ddgs.text(query, region='wt-wt', safesearch='Off', timelimit='y')
                for r in islice(response, 20):
                    result.append({'title': r['title'], 'url': r['href']})
                return result

        except requests.RequestException as e:
            # Handle request exceptions.
            print(f"Attempt {attempt + 1} raised an error: {e}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(1)

        except Exception as e:
            # Handle other exceptions.
            print(f"An unexpected error occurred on attempt {attempt + 1}: {e}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(1)

    else:
        # If max retries reached, exit the function.
        print("Max retries reached. Exiting...")
        return None

def scrape_website(objective: str, url: str):
    """
    Scrape and potentially summarize the content of a website based on a given objective.
    Args:
    - objective (str): The objective & task that users give to the agent.
    - url (str): The URL of the website to be scraped.
    Returns:
    - str: Extracted or summarized content of the website.
    """
    print("Scraping website...")
    try:
        # Use NewsPlease to scrape the website.
        article = NewsPlease.from_url(url)
        print(f'{article.title} - {article.url}')
        text = article.maintext
        # Summarize if content is too large.
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    except:
        pass

def summary(objective, content):
    """
    Generate a summary for a given content based on the objective.
    Args:
    - objective (str): The objective for the summary.
    - content (str): The content to be summarized.
    Returns:
    - str: Summarized content.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", streaming=True)

    # Split the content into manageable chunks.
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])

    # Load the summary chain with necessary configurations.
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template, combine_prompt=map_prompt_template, verbose=True)

    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website function."""
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    """
    A tool that provides functionality to scrape a website.
    """
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        """Runs the scrape_website function."""
        return scrape_website(objective, url)

    def _arun(self, url: str):
        """Asynchronous version of _run. (Currently not implemented)"""
        raise NotImplementedError("error here")

@cl.langchain_factory(use_async=False)
def run():
    """
    Initialize and return a langchain agent with search and scraping tools.
    Returns:
    - Agent: Initialized langchain agent.
    """
    tools = [
        Tool(name="Search", func=search, description="useful for when you need to answer questions about current events, data. You should ask targeted questions"),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
                you do not make things up, you will try as hard as possible to gather facts & data to back up the research
                
                Please make sure you complete the objective above with the following rules:
                1/ You should do enough research to gather as much information as possible about the objective
                2/ If there are url of relevant links & articles, you will scrape it to gather more information
                3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
                4/ You should not make things up, you should only write facts & data that you have gathered
                5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
                6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
    )
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    # Initialize the ChatOpenAI model.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", streaming=True)
    memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm)

    # Initialize the agent with tools and other configurations.
    return initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, agent_kwargs=agent_kwargs, memory=memory)
