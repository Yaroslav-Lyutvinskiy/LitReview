from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from semanticscholar import SemanticScholar
from markitdown import MarkItDown
import requests
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import arxiv
import ast 

from dataclasses import dataclass
from pydantic import BaseModel, Field 
from typing import List, Optional, Any
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, SingleThreadedAgentRuntime
from autogen_agentchat.messages import TextMessage
import asyncio 
import time
from openai._exceptions import RateLimitError


def download_pdf_file(url: str) -> str:
    """
    Download PDF from given URL to local directory.
    :param url: The url of the PDF file to be downloaded
    :return: path of the downloaded file, empty string if download failed
    """

    # Request URL and get response object
    response = requests.get(url, stream=True)

    # isolate PDF filename from URL
    pdf_file_name = os.path.basename(url)
    if response.status_code == 200:
        # Save in current working directory
        filepath = os.path.join(os.getcwd(), pdf_file_name)
        if not filepath.endswith(".pdf"):
            filepath = filepath+".pdf"
        with open(filepath, 'wb') as pdf_object:
            pdf_object.write(response.content)
            print(f'{pdf_file_name} was successfully saved!')
            return filepath
    else:
        print(f'Could not download {pdf_file_name},')
        print(f'HTTP response status code: {response.status_code}')
        return None


@dataclass
class Article(BaseModel):
    title:str
    authors:List[str]
    published:str
    abstract:Optional[str] = None
    full_text_url:Optional[str] = None
    full_text:Optional[str] = None
    summary:Optional[str] = None
    topic:Optional[str] = None
    def __init__(self, title, authors, published, abstract, full_text_url, full_text = None, topic=None, summary = None):
        super().__init__(title = title, authors = authors, published = published, abstract = abstract, full_text_url = full_text_url, full_text = full_text,  topic = topic, summary = summary)

@dataclass
class Articles(BaseModel):
    articles:str
    def __init__(self, articles:str):
        super().__init__(articles = articles)
    

@dataclass
class UserRequest(BaseModel):
    request:str
    def __init__(self, request:str):
        super().__init__(request = request)



def arxiv_search(query: str, max_results: int = 2) -> list[Article]:  # type: ignore[type-arg]
    """
    Search Arxiv for papers and return the results including abstracts and full text.
    :param query: query to search in arxiv
    :param max_results: maximum number of articles returned
    :return: list of found papers
    """

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=5,
        num_retries=3
    )
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    
    for paper in client.results(search):

        article =  Article(
            title = paper.title, 
            authors = [author.name for author in paper.authors],
            published = paper.published.strftime("%Y-%m-%d"),
            abstract = paper.summary,
            full_text_url = paper.pdf_url,
            summary = None
        )
        results.append(article)

    return results

class ArxivSearchAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        global model_client
        self._delegate = AssistantAgent(
            name, 
            model_client=model_client,
            tools=[arxiv_search_tool],
            description="An agent that can search Arxiv for papers related to a given topic, including abstracts and full text",
            system_message="You are a helpful AI assistant. Solve tasks using your tools. Specifically, you can take into consideration the user's request and craft a search query that is most likely to return relevant academic papers.",
        )

    @message_handler
    async def handle_my_message_type(self, message: UserRequest, ctx: MessageContext) -> Any:
        response = await self._delegate.on_messages(
            [TextMessage(content=message.request, source="user")], 
            ctx.cancellation_token
        )
        
        s = response.chat_message.results[0].content
        safe_globals = {"Article": Article}
        articles = eval(s, safe_globals, {})    
        
        counter = 0 
        waities = []
        #loop = asyncio.get_event_loop()

        finished = []
        for article in articles:
            agent_name = "Summarizer"+str(counter)
            try:
                agent = await runtime.agent_metadata(AgentId.from_str(agent_name+"/default"))
            except:
                await SummarizerAgent.register(runtime, agent_name, lambda: SummarizerAgent(agent_name))
            #finished.append(await  self.send_message(article, AgentId(agent_name, "default")))
            waities.append(asyncio.create_task(self.send_message(article,AgentId(agent_name, "default"))))
            
        finished, unfinished = await asyncio.wait(waities, timeout=1200)
        print(finished)

        #processing of finished tasks
        articles_with_summary = []
        for task in list(finished):
            if (task.done()  and task.exception() is None):
                articles_with_summary.append(task.result())

        if len(articles_with_summary) == 0:
            return "No articles available"

        report = await self.send_message(Articles(articles=str(articles_with_summary)),AgentId("ReporterAgent", "default"))

        return report


class SummarizerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        global model_client
        self._delegate = AssistantAgent(
            name, 
            model_client=model_client,
            description="An agent summarizes a scientific paper. The paper is provided as title, list of authors, abstaract and full text. Summarization is to be builtin context of provided user query",
            system_message="You are a helpful AI assistant. Summarize content of scientific paper provided in no more than 1500 words. For summary use literature review style. Keep key references and provide references mentioned in the summary in correct format in last 'References' section" +
            "The paper is provided as title, list of authors, abstaract and full text.",
       )

    @message_handler
    async def handle_article(self, message: Article, ctx: MessageContext) -> Article:
        pdf_file = None
        try:
            #extract full text
            md_result = None
            if message.full_text_url is not None : 
                pdf_file =  download_pdf_file(message.full_text_url)
                if pdf_file is not None : 
                    md = MarkItDown()
                    md_result = md.convert(pdf_file).markdown
                    message.full_text = md_result

            counter = 0; 
            while True:
                try:
                    response = await self._delegate.on_messages(
                        [TextMessage(content=str(message), source="Summarizer")], 
                        ctx.cancellation_token
                    )
                    counter += 1
                except RateLimitError as error:
                    print(f"Rate Limit on {message.full_text_url}, error - : {error}")
                    ## Yes it is not mistake - we need to sleep everything to acheve reset of RateLimit on OpenAI
                    await asyncio.sleep(30+60*counter)
                except:
                    pass
                else:
                    print(f"Article {message.full_text_url} has been processed")
                    break
            
            message.summary = response.chat_message.content
            message.full_text = None
        # except:
        finally:
            if pdf_file is not None:
                os.remove(pdf_file)
        return message
    

class ReporterAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        global model_client
        self._delegate = AssistantAgent(
            name, 
            model_client=model_client,
            description="Generate a report based on a given topic",
            system_message="You are a helpful assistant. Your task is to synthesize data extracted into a high quality literature review containing no more than 5000 words including CORRECT references. "+
                "You MUST write a final report that is formatted as a literature review with CORRECT references.",
       )

   
    @message_handler
    async def handle_articles(self, message: Articles, ctx: MessageContext) -> Any:
        counter = 0; 
        while True:
            try:
                counter += 1
                response = await self._delegate.on_messages(
                    [TextMessage(content=message.articles, source="Reporter")], 
                    ctx.cancellation_token)
            except RateLimitError as error:
                print(f"Rate Limit on report, error - : {error}")
                await asyncio.sleep(30+60*counter)
            except:
                pass
            else:
                break
        
        return response 



model_client = None
runtime = None

def init_team(model_client_param):

    global model_client 
    model_client = model_client_param

    global runtime
    runtime = SingleThreadedAgentRuntime()
    await ArxivSearchAgent.register(runtime, "SearchAgent", lambda: ArxivSearchAgent("SearchAgent"))
    await ReporterAgent.register(runtime, "ReporterAgent", lambda: ReporterAgent("ReporterAgent"))


    return 


async def run_team(query):
    
    response = await runtime.send_message(
        UserRequest(request = query), AgentId("SearchAgent", "default"))

    return {"message": response.chat_message.content}

