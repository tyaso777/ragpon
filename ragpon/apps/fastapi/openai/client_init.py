import json
import os
from typing import Any, AsyncGenerator, Generator, Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai._exceptions import OpenAIError
from openai.types.chat import ChatCompletion

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)

ModelType = Literal["openai", "azure"]


def create_openai_client():
    """
    Reads environment variables, decides whether to use Azure or regular OpenAI,
    and returns the client plus associated model/deployment info.

    Returns:
        tuple: (client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE)
    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load environment variables
    OPENAI_TYPE = os.getenv("OPENAI_TYPE", "openai").lower()  # Default is "openai"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
        model_name = AZURE_OPENAI_MODEL
        deployment_id = AZURE_OPENAI_DEPLOYMENT
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        model_name = OPENAI_MODEL
        deployment_id = None

    return client, model_name, deployment_id, OPENAI_TYPE


def create_async_openai_client():
    """
    Initializes an async OpenAI or AzureOpenAI client from environment variables.

    Returns:
        tuple: (async_client, model_name, deployment_id, openai_type)
    """
    OPENAI_TYPE = os.getenv("OPENAI_TYPE", "openai").lower()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if OPENAI_TYPE == "azure":
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
        }
    else:
        required_vars = {"OPENAI_API_KEY": OPENAI_API_KEY}

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    if OPENAI_TYPE == "azure":
        client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        model_name = AZURE_OPENAI_MODEL
        deployment_id = AZURE_OPENAI_DEPLOYMENT
    else:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        model_name = OPENAI_MODEL
        deployment_id = None

    return client, model_name, deployment_id, OPENAI_TYPE


def call_llm_sync_with_handling(
    *,
    client: OpenAI | AzureOpenAI,
    model: str,
    messages: list[dict[str, str]],
    user_id: str,
    session_id: str,
    temperature: float = 0.7,
    stream: bool = False,
    model_type: ModelType = "openai",
    **kwargs: Any,
) -> ChatCompletion | Generator[str, None, None]:
    """
    Synchronous version of the LLM call function with error handling.

    Args:
        client (OpenAI | AzureOpenAI): The synchronous OpenAI-compatible client.
        model (str): Model name or deployment ID.
        messages (list[dict[str, str]]): Chat history.
        user_id (str): For logging.
        session_id (str): For logging.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        stream (bool, optional): Whether to stream response. Defaults to False.
        model_type (Literal["openai", "azure"], optional): Backend type.
        **kwargs: Additional parameters.

    Returns:
        ChatCompletion | Generator[str, None, None]: Full response or streamed content chunks.

    Raises:
        ValueError: Raised when OpenAI API call fails, JSON decoding fails, or an unexpected error occurs.
    """
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages
        kwargs["temperature"] = temperature
        kwargs["stream"] = stream

        if stream:
            logger.info(
                f"[call_llm_sync_with_handling] Streaming LLM call initiated: "
                f"user_id={user_id}, session_id={session_id}, model={model}, model_type={model_type}"
            )

            def stream_generator() -> Generator[str, None, None]:
                response = client.chat.completions.create(**kwargs)
                for chunk in response:
                    try:
                        choices = chunk.choices
                        if not choices:
                            continue
                        delta = choices[0].delta
                        finish_reason = choices[0].finish_reason

                        if delta.content is None and finish_reason == "stop":
                            continue

                        if delta.content:
                            yield delta.content
                        else:
                            logger.debug(
                                f"[call_llm_sync_with_handling] Unexpected empty chunk: "
                                f"user_id={user_id}, session_id={session_id}, chunk={chunk}"
                            )
                    except (IndexError, AttributeError) as e:
                        logger.warning(
                            f"[call_llm_sync_with_handling] Malformed chunk: {e}; chunk={chunk}"
                        )

            return stream_generator()
        else:
            logger.info(
                f"[call_llm_sync_with_handling] Non-streaming LLM call initiated: "
                f"user_id={user_id}, session_id={session_id}, model={model}, model_type={model_type}"
            )
            response = client.chat.completions.create(**kwargs)
            return response

    except OpenAIError as e:
        logger.exception("[call_llm_sync_with_handling] OpenAI API error")
        raise ValueError("OpenAI API failed") from e
    except json.JSONDecodeError as e:
        logger.exception("[call_llm_sync_with_handling] JSON decode error")
        raise ValueError("Invalid JSON response from model") from e
    except Exception as e:
        logger.exception("[call_llm_sync_with_handling] Unexpected error")
        raise ValueError("Unexpected error occurred during LLM call") from e


async def call_llm_async_with_handling(
    *,
    client: AsyncOpenAI | AsyncAzureOpenAI,
    model: str,
    messages: list[dict[str, str]],
    user_id: str,
    session_id: str,
    temperature: float = 0.7,
    stream: bool = False,
    model_type: ModelType = "openai",
    **kwargs: Any,
) -> ChatCompletion | AsyncGenerator[str, None]:
    """
    Calls OpenAI or Azure OpenAI ChatCompletion API with unified error handling and optional streaming.

    Args:
        client (AsyncOpenAI | AsyncAzureOpenAI): The OpenAI-compatible async client.
        model (str): The model name or Azure deployment ID.
        messages (list[dict[str, str]]): The chat history for the request.
        user_id (str): The user ID for logging and traceability.
        session_id (str): The session ID for logging and traceability.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        stream (bool, optional): Whether to enable streaming response. Defaults to False.
        model_type (Literal["openai", "azure"], optional): Backend type. Defaults to "openai".
        **kwargs: Additional keyword arguments passed to the ChatCompletion API.

    Returns:
        ChatCompletion | AsyncGenerator[str, None]: The chat response as a full result object,
        or an async generator yielding streamed content chunks.

    Raises:
        ValueError: Raised when OpenAI API call fails, JSON decoding fails, or an unexpected error occurs.
    """
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages
        kwargs["temperature"] = temperature
        kwargs["stream"] = stream

        if stream:
            # Streaming generator
            async def stream_generator() -> AsyncGenerator[str, None]:
                response = await client.chat.completions.create(**kwargs)
                async for chunk in response:
                    try:
                        choices = chunk.choices
                        if not choices:
                            continue
                        delta = choices[0].delta
                        finish_reason = choices[0].finish_reason

                        # Skip harmless stop chunk without warning
                        if delta.content is None and finish_reason == "stop":
                            continue

                        if delta.content:
                            yield delta.content

                        else:
                            # Real issue: unexpected None
                            logger.warning(
                                f"[call_llm_with_handling] Unexpected empty chunk: "
                                f"user_id={user_id}, session_id={session_id}, chunk={chunk}"
                            )

                    except (IndexError, AttributeError) as e:
                        logger.warning(
                            f"[call_llm_with_handling] Malformed chunk: {e}; chunk={chunk}"
                        )

            return stream_generator()
        else:
            response = await client.chat.completions.create(**kwargs)
            return response

    except OpenAIError as e:
        logger.exception("[call_llm_with_handling] OpenAI API error")
        raise ValueError("OpenAI API failed") from e
    except json.JSONDecodeError as e:
        logger.exception("[call_llm_with_handling] JSON decode error")
        raise ValueError("Invalid JSON response from model") from e
    except Exception as e:
        logger.exception("[call_llm_with_handling] Unexpected error")
        raise ValueError("Unexpected error occurred during LLM call") from e
