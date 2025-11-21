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


def s2_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Search Semantic scholar by keywords and return the results including abstracts.
    """
    from semanticscholar import SemanticScholar

    sch = SemanticScholar()
    
    search = sch.search_paper(query=query, limit = max_results)

    results = []
    for paper in search.items:
        try:
            results.append(
                {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.publicationDate.strftime("%Y-%m-%d"),
                    "abstract": paper.abstract,
                    "pdf_url": paper.openAccessPdf["url"]
                }
            )
        except:
            continue

    # # Write results to a file
    # with open('s2_search_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)

    return results

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


#def arxiv_search(query: str, max_results: int = 2, min_year: int = 2020, max_year: int = 2025) -> list:  # type: ignore[type-arg]
async def arxiv_search(query: str, max_results: int = 2) -> list:  # type: ignore[type-arg]
    """
    Asynchronious function to search Arxiv for papers and return the results including abstracts and summary.
    :param query: query to search in arxiv
    :param max_results: maximum number of articles returned
    :return: path to output markdown file
    """

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=5,
        num_retries=3
    )
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    summarizer_agent = AssistantAgent(
        name="Summarizer_Agent",
        model_client=model_client,
        description="An agent that can summarize one scientific paper in a time. The paper is provided as title, list of authors, abstaract and full text. Summarization is to be builtin context of provided user query",
        system_message="You are a helpful AI assistant. Summarize content of scientific paper provided in no more that 2000 words. For summary use literature review style. Provide references mentioned in the summary in correct format" +
        "The paper is provided as title, list of authors, abstaract and full text. Summarization is to be built in context of provided user query",
    )

    results = []
    for paper in client.results(search):

        try:

            #extract full text
            md_result = None
            if paper.pdf_url is not None : 
                pdf_file =  download_pdf_file(paper.pdf_url)
                if pdf_file is not None : 
                    md = MarkItDown()
                    md_result = md.convert(pdf_file).markdown
                
            
            paper = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d"),
                "abstract": paper.summary,
                "full_text":md_result,
                "query": query    
            }

            paper_summary = await summarizer_agent.run(
                task = f"Write a summary of provided article : {paper}",
            )
            paper["summary"] = paper_summary.messages[-1].content
            paper["full_text"] = None
            await summarizer_agent.on_reset(None)

            results.append(paper)
            
        # except:
        #     # Here need to be some warning
        #     pass
        finally:
            if pdf_file is not None:
                os.remove(pdf_file)

    # # Write results to a file
    # with open('arxiv_search_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)

    return results


model_client = None

def init_team(model_client_param):

    global model_client 
    model_client = model_client_param

    arxiv_search_tool = FunctionTool(
        arxiv_search, description="Search Arxiv for papers related to a given topic, including abstracts and summary"
    )

    arxiv_search_agent = AssistantAgent(
        name="Arxiv_Search_Agent",
        tools=[arxiv_search_tool],
        model_client=model_client,
        description="An agent that can search Arxiv for papers related to a given topic, including abstracts and summary",
        system_message="You are a helpful AI assistant. Given the user input define the search topic for scientific articles in arxiv, and search arxive for the topic. return list of found articles and topic",
    )

    report_agent = AssistantAgent(
        name="Report_Agent",
        model_client=model_client,
        description="Generate a report based on a given topic",
        system_message="You are a helpful assistant. Your task is to synthesize data extracted into a high quality literature review containing no more than 5000 words including CORRECT references. "+
            "You MUST write a final report that is formatted as a literature review with CORRECT references. Your response should end with the word 'TERMINATE'"
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([arxiv_search_agent, report_agent], termination_condition=termination)

    return team


async def run_team(team,query):
    
    stream = team.run_stream(task=query)

    async for message in stream:
        pass

    return {"message": message.messages[-1].content}

