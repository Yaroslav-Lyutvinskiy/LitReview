import sys

from fastapi import FastAPI
from dotenv import load_dotenv
import os

from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv("./.env")
openai_api_key = os.environ.get("OPENAI_API_KEY")

from app.agents import init_team, run_team

model_client = OpenAIChatCompletionClient(
    model="o3-mini",
    api_key = openai_api_key
)


version = f"{sys.version_info.major}.{sys.version_info.minor}"

app = FastAPI()


@app.get("/")
async def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn. Using Python {version}"
    return {"message": message}

@app.on_event("startup")
def startup_event():
    init_team(model_client)

@app.get("/agent/")
async def agent_call(query):

    response = await run_team(query)

    return response


