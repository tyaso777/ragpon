import json
import os
from typing import Any, AsyncGenerator, Generator, Literal

from openai import AsyncOpenAI, OpenAI
from openai._exceptions import OpenAIError
from openai.types.chat import ChatCompletion

from ragpon._utils.logging_helper import get_library_logger

logger = get_library_logger(__name__)

ModelType = Literal["openai", "azure"]
ApiMode = Literal["chat_completions", "responses"]


def _normalize_azure_v1_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return f"{endpoint}/"
    if endpoint.endswith("/openai"):
        return f"{endpoint}/v1/"
    return f"{endpoint}/openai/v1/"


def _transform_tools_for_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            transformed.append(tool)
            continue

        function_def = tool.get("function", {})
        transformed.append(
            {
                "type": "function",
                "name": function_def["name"],
                "description": function_def.get("description"),
                "parameters": function_def["parameters"],
                "strict": True,
            }
        )
    return transformed


def _transform_tool_choice_for_responses(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    function_def = tool_choice.get("function", {})
    if "name" not in function_def:
        return tool_choice
    return {"type": "function", "name": function_def["name"]}


def _prepare_responses_request_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    stream: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": messages,
        "stream": stream,
        "temperature": temperature,
    }

    for key, value in kwargs.items():
        if key == "max_tokens":
            payload["max_output_tokens"] = value
        elif key == "tools" and isinstance(value, list):
            payload["tools"] = _transform_tools_for_responses(value)
        elif key == "tool_choice":
            payload["tool_choice"] = _transform_tool_choice_for_responses(value)
        elif key in {
            "top_p",
            "max_output_tokens",
            "reasoning",
            "parallel_tool_calls",
            "metadata",
        }:
            payload[key] = value

    return payload


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = getattr(response, "output", None) or []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for content_item in getattr(item, "content", None) or []:
            if getattr(content_item, "type", None) == "output_text":
                text = getattr(content_item, "text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise ValueError("LLM response missing expected text output")


def extract_text_from_llm_response(response: Any) -> str:
    if hasattr(response, "choices"):
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise ValueError("LLM response missing expected chat content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response returned empty text content")
        return content.strip()

    return _extract_response_text(response)


def extract_tool_calls_from_llm_response(response: Any) -> list[dict[str, str]]:
    if hasattr(response, "choices"):
        try:
            tool_calls = response.choices[0].message.tool_calls or []
        except (IndexError, AttributeError) as exc:
            raise ValueError("LLM response missing choices/tool_calls") from exc

        return [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in tool_calls
        ]

    tool_calls_out: list[dict[str, str]] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        tool_calls_out.append(
            {
                "id": getattr(item, "call_id", getattr(item, "id", "")),
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", ""),
            }
        )
    return tool_calls_out


def create_openai_client():
    openai_type = os.getenv("OPENAI_TYPE", "openai").lower()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.1")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if openai_type == "azure":
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": azure_endpoint,
            "AZURE_OPENAI_API_KEY": azure_api_key,
            "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
        }
    else:
        required_vars = {"OPENAI_API_KEY": openai_api_key}

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    if openai_type == "azure":
        client = OpenAI(
            api_key=azure_api_key,
            base_url=_normalize_azure_v1_base_url(azure_endpoint),
        )
        model_name = azure_model
        deployment_id = azure_deployment
        api_mode: ApiMode = "responses"
    else:
        client = OpenAI(api_key=openai_api_key)
        model_name = openai_model
        deployment_id = None
        api_mode = "chat_completions"

    return client, model_name, deployment_id, openai_type, api_mode


def create_async_openai_client():
    openai_type = os.getenv("OPENAI_TYPE", "openai").lower()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.1")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if openai_type == "azure":
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": azure_endpoint,
            "AZURE_OPENAI_API_KEY": azure_api_key,
            "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
        }
    else:
        required_vars = {"OPENAI_API_KEY": openai_api_key}

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    if openai_type == "azure":
        client = AsyncOpenAI(
            api_key=azure_api_key,
            base_url=_normalize_azure_v1_base_url(azure_endpoint),
        )
        model_name = azure_model
        deployment_id = azure_deployment
        api_mode: ApiMode = "responses"
    else:
        client = AsyncOpenAI(api_key=openai_api_key)
        model_name = openai_model
        deployment_id = None
        api_mode = "chat_completions"

    return client, model_name, deployment_id, openai_type, api_mode


def call_llm_sync_with_handling(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    user_id: str,
    session_id: str,
    temperature: float = 0.7,
    stream: bool = False,
    model_type: ModelType = "openai",
    api_mode: ApiMode = "chat_completions",
    **kwargs: Any,
) -> ChatCompletion | Generator[str, None, None] | Any:
    try:
        if api_mode == "responses":
            request_kwargs = _prepare_responses_request_kwargs(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                kwargs=kwargs,
            )
        else:
            request_kwargs = dict(kwargs)
            request_kwargs["model"] = model
            request_kwargs["messages"] = messages
            request_kwargs["temperature"] = temperature
            request_kwargs["stream"] = stream

        if stream:
            logger.info(
                "[call_llm_sync_with_handling] Streaming LLM call initiated: "
                "user_id=%s, session_id=%s, model=%s, model_type=%s, api_mode=%s",
                user_id,
                session_id,
                model,
                model_type,
                api_mode,
            )

            def stream_generator() -> Generator[str, None, None]:
                if api_mode == "responses":
                    response = client.responses.create(**request_kwargs)
                    for event in response:
                        if getattr(event, "type", None) == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                yield delta
                    return

                response = client.chat.completions.create(**request_kwargs)
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
                    except (IndexError, AttributeError) as exc:
                        logger.warning(
                            "[call_llm_sync_with_handling] Malformed chunk: %s; chunk=%s",
                            exc,
                            chunk,
                        )

            return stream_generator()

        logger.info(
            "[call_llm_sync_with_handling] Non-streaming LLM call initiated: "
            "user_id=%s, session_id=%s, model=%s, model_type=%s, api_mode=%s",
            user_id,
            session_id,
            model,
            model_type,
            api_mode,
        )
        if api_mode == "responses":
            return client.responses.create(**request_kwargs)
        return client.chat.completions.create(**request_kwargs)

    except OpenAIError as exc:
        logger.exception("[call_llm_sync_with_handling] OpenAI API error")
        raise ValueError("OpenAI API failed") from exc
    except json.JSONDecodeError as exc:
        logger.exception("[call_llm_sync_with_handling] JSON decode error")
        raise ValueError("Invalid JSON response from model") from exc
    except Exception as exc:
        logger.exception("[call_llm_sync_with_handling] Unexpected error")
        raise ValueError("Unexpected error occurred during LLM call") from exc


async def call_llm_async_with_handling(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    user_id: str,
    session_id: str,
    temperature: float = 0.7,
    stream: bool = False,
    model_type: ModelType = "openai",
    api_mode: ApiMode = "chat_completions",
    **kwargs: Any,
) -> ChatCompletion | AsyncGenerator[str, None] | Any:
    try:
        if api_mode == "responses":
            request_kwargs = _prepare_responses_request_kwargs(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                kwargs=kwargs,
            )
        else:
            request_kwargs = dict(kwargs)
            request_kwargs["model"] = model
            request_kwargs["messages"] = messages
            request_kwargs["temperature"] = temperature
            request_kwargs["stream"] = stream

        if stream:

            async def stream_generator() -> AsyncGenerator[str, None]:
                if api_mode == "responses":
                    response = await client.responses.create(**request_kwargs)
                    async for event in response:
                        if getattr(event, "type", None) == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                yield delta
                    return

                response = await client.chat.completions.create(**request_kwargs)
                async for chunk in response:
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
                    except (IndexError, AttributeError) as exc:
                        logger.warning(
                            "[call_llm_with_handling] Malformed chunk: %s; chunk=%s",
                            exc,
                            chunk,
                        )

            return stream_generator()

        if api_mode == "responses":
            return await client.responses.create(**request_kwargs)
        return await client.chat.completions.create(**request_kwargs)

    except OpenAIError as exc:
        logger.exception("[call_llm_with_handling] OpenAI API error")
        raise ValueError("OpenAI API failed") from exc
    except json.JSONDecodeError as exc:
        logger.exception("[call_llm_with_handling] JSON decode error")
        raise ValueError("Invalid JSON response from model") from exc
    except Exception as exc:
        logger.exception("[call_llm_with_handling] Unexpected error")
        raise ValueError("Unexpected error occurred during LLM call") from exc
