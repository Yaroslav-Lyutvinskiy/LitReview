import sys

from fastapi import FastAPI

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

api_key = "<YOUR_API_KEY>"
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key = api_key
)

version = f"{sys.version_info.major}.{sys.version_info.minor}"

app = FastAPI()


@app.get("/")
async def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn. Using Python {version}"
    return {"message": message}


@app.get("/agent/")
async def agent_call(query):

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    team = MagenticOneGroupChat([surfer], model_client=model_client)

    stream = team.run_stream(task=query)

    async for message in stream:
        pass

    return {"message": message.messages[-1].content}
