# %%
# FastAPI side
import os
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AzureOpenAI, OpenAI

app = FastAPI()

# Load environment variables
OPENAI_TYPE = os.getenv("OPENAI_TYPE", "openai").lower()  # Default is "openai"

# Settings for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Settings for Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Check required environment variables
if OPENAI_TYPE == "azure":
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
    }
else:
    required_vars = {"OPENAI_API_KEY": OPENAI_API_KEY}

missing_vars = [key for key, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Initialize the client
if OPENAI_TYPE == "azure":
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    MODEL_NAME = AZURE_OPENAI_MODEL
    DEPLOYMENT_ID = AZURE_OPENAI_DEPLOYMENT
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = OPENAI_MODEL
    DEPLOYMENT_ID = None


def stream_chat_completion(
    user_query: str, contexts: list
) -> Generator[str, None, None]:
    # (Same as before)
    try:
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": f"Context: {contexts}"},
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.7,
            "stream": True,
        }
        if OPENAI_TYPE == "azure":
            kwargs["deployment_id"] = DEPLOYMENT_ID

        response = client.chat.completions.create(**kwargs)

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield f"data: {content}\n\n"

    except Exception as e:
        yield f"data: Error during streaming: {str(e)}\n\n"


@app.post("/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries")
async def handle_query(user_id: str, app_name: str, session_id: str, request: Request):
    # (Same as before)
    try:
        data = await request.json()
        user_query = data.get("query", "")
        file_info = data.get("file", None)
        is_private = data.get("is_private_session", False)
    except Exception as e:
        return StreamingResponse(
            iter([f"data: Error parsing request: {str(e)}\n\n"]),
            media_type="text/event-stream",
        )

    retrieved_contexts = ["Some retrieved context 1", "Some retrieved context 2"]
    return StreamingResponse(
        stream_chat_completion(user_query, retrieved_contexts),
        media_type="text/event-stream",
    )
