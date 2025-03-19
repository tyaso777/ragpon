# %%
# FastAPI side
import json
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ragpon import (
    BaseDocument,
    BM25Repository,
    ChromaDBEmbeddingAdapter,
    ChromaDBRepository,
    Config,
    Document,
    RuriLargeEmbedderCTranslate2,
)
from ragpon._utils.logging_helper import get_library_logger
from ragpon.api.client_init import create_openai_client
from ragpon.tokenizer import SudachiTokenizer

app = FastAPI()

config = Config(
    config_file=r"D:\Users\AtsushiSuzuki\OneDrive\デスクトップ\test\ragpon\ragpon\examples\sample_config.yml"
)
config.set(
    "DATABASES.BM25_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db\\bm25",
)
config.set(
    "DATABASES.CHROMADB_FOLDER_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db",
)

embedder = ChromaDBEmbeddingAdapter(RuriLargeEmbedderCTranslate2(config=config))

# Initialize logger
logger = get_library_logger(__name__)

# import logging

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )

chroma_repo = ChromaDBRepository(
    collection_name="pdf_collection",
    embed_func=embedder,
    metadata_class=BaseDocument,
    result_class=Document,
    similarity="cosine",
    connection_mode="http",
    folder_path=None,
    http_url="localhost",
    port=8007,
)

bm25_repo = BM25Repository(
    db_path=config.get("DATABASES.BM25_PATH"),
    schema=BaseDocument,
    result_class=Document,
    tokenizer=SudachiTokenizer(),
)

client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE = create_openai_client()


def mock_insert_three_records(
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    user_query: str,
    retrieved_contexts_str: str,
    total_stream_content: str,
    llm_model: str,
    rerank_model: str | None,
) -> None:
    """
    Mocks inserting three records (user/system/assistant) into the DB.
    In reality, you'd do a single transaction with three INSERT statements
    or something similar.

    Args:
        user_id (str): The user ID.
        session_id (str): The session ID.
        app_name (str): The app name.
        round_id (int): The round number.
        user_msg_id (str): ID for the user message.
        system_msg_id (str): ID for the system message (if any).
        assistant_msg_id (str): ID for the assistant message.
        user_query (str): The user's query content.
        retrieved_contexts_str (str): The retrieved context string from RAG.
        total_stream_content (str): The complete assistant's streamed response.
        llm_model (str): The model name used (e.g., GPT-4).
        rerank_model (str | None): The reranker model, if any (None here).
    """
    logger.info("[MOCK INSERT] Storing 3 records in the DB with the following data:")
    logger.info(
        "  user_id=%s, session_id=%s, app_name=%s, round_id=%d",
        user_id,
        session_id,
        app_name,
        round_id,
    )
    logger.info(
        "  user_msg_id=%s, system_msg_id=%s, assistant_msg_id=%s",
        user_msg_id,
        system_msg_id,
        assistant_msg_id,
    )
    logger.info("  user_query=%s", user_query)
    # logger.info("  retrieved_contexts_str=%s", retrieved_contexts_str)
    logger.info("  total_stream_content=%s", total_stream_content)
    logger.info("  llm_model=%s, rerank_model=%s", llm_model, rerank_model)


def generate_queries_from_history(
    messages_list: list[dict[str, str]], system_instructions: str
) -> list[str]:
    """
    Generate multiple search queries from a given conversation history.

    This function takes the existing conversation (messages_list) and appends
    final instructions (system_instructions) to request a JSON array of objects
    like: [{"query": "..."}]. The model's response is parsed into Python objects,
    and a list of query strings is returned.

    Args:
        messages_list (list[dict[str, str]]):
            The conversation so far, e.g.:
            [
              {"role": "user", "content": "User's question 1"},
              {"role": "assistant", "content": "Assistant's answer 1"},
              ...
            ]
        system_instructions (str):
            The final system message that instructs the model to produce
            multiple queries in JSON format, for example:
            "You are a helpful assistant. Return an array of objects with
             a 'query' key in each."

    Returns:
        list[str]: A list of query strings extracted from the model's JSON output.

    Raises:
        ValueError: If the JSON returned by the model is malformed or missing
            the expected structure.
    """
    # 1) Build the messages for the ChatCompletion request
    #    We'll append a final system message at the end
    final_messages = messages_list + [
        {"role": "system", "content": system_instructions}
    ]

    # 2) Prepare ChatCompletion arguments
    kwargs = {
        "model": MODEL_NAME,
        "messages": final_messages,
        "temperature": 0.0,  # Lower temperature => more deterministic JSON
    }

    if OPENAI_TYPE == "azure" and DEPLOYMENT_ID is not None:
        kwargs["deployment_id"] = DEPLOYMENT_ID

    # 3) Call the OpenAI ChatCompletion API
    response = client.chat.completions.create(**kwargs)
    raw_text = response.choices[0].message.content.strip()

    # 4) Parse the JSON output (expected: [{"query": "..."}])
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model: {raw_text}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got: {type(parsed)}")

    # 5) Collect queries
    queries = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict at index {i}, got: {type(item)}")
        if "query" not in item:
            raise ValueError(f"Missing 'query' key at index {i} in: {item}")
        queries.append(item["query"])

    return queries


def stream_chat_completion(
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    messages: list[dict],
    retrieved_contexts_str: str,
) -> Generator[str, None, None]:
    """
    Generates a streaming response for the user's query by calling the LLM, then
    inserts three records into a mock database once the streaming completes.

    This function yields chunks of text to the client via Server-Sent Events (SSE)
    and accumulates the entire assistant response locally. After the final chunk
    is yielded, it performs a mock database insertion for user/system/assistant
    messages, including relevant metadata.

    Args:
        user_id (str): The ID of the user initiating the query.
        session_id (str): The ID of the session associated with this conversation.
        app_name (str): The name of the application or service.
        round_id (int): The conversation round number (used to group messages).
        user_msg_id (str): Unique identifier for the user's message.
        system_msg_id (str): Unique identifier for the system message.
        assistant_msg_id (str): Unique identifier for the assistant's message.
        messages (list[dict]): A list of messages in the format:
            [ { "role": "user"/"assistant"/"system", "content": "..." }, ... ]
        retrieved_contexts_str (str): A string containing retrieved context data
            (e.g., from RAG search) to be provided to the LLM.

    Yields:
        str: SSE data chunks in the format "data: ...\\n\\n" for each partial response
        from the LLM. Also yields a final "[DONE]" marker if streaming completes
        successfully.

    Raises:
        Exception: Propagates any exceptions encountered during the LLM streaming
        process, yielding an error message to the SSE client.
    """
    accumulated_content = ""  # Will hold the entire assistant response
    rerank_model = None  # If you want to pass something else, do so

    try:
        # 1) Build the OpenAI messages array
        #    We can prepend system instructions + RAG context, then append the conversation
        openai_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. In your responses, please "
                    "include the file names and page numbers that serve as "
                    "the basis for your statements..."
                ),
            },
            {
                "role": "system",
                "content": f"Context: {retrieved_contexts_str}",
            },
        ]
        # Now extend with each item in 'messages'
        # (We assume each item in messages is already {"role": ..., "content": ...})
        openai_messages.extend(messages)

        kwargs = {
            "model": MODEL_NAME,
            "messages": openai_messages,
            "temperature": 0.7,
            "stream": True,
        }
        if OPENAI_TYPE == "azure":
            kwargs["deployment_id"] = DEPLOYMENT_ID

        response = client.chat.completions.create(**kwargs)

        # (A) Yield chunks to the client and accumulate them
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                accumulated_content += content
                yield f"data: {content}\n\n"

        user_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        # (B) Once streaming is finished, do the mock DB insert
        mock_insert_three_records(
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
            round_id=round_id,
            user_msg_id=user_msg_id,
            system_msg_id=system_msg_id,
            assistant_msg_id=assistant_msg_id,
            user_query=user_query,
            retrieved_contexts_str=retrieved_contexts_str,
            total_stream_content=accumulated_content,
            llm_model=MODEL_NAME,
            rerank_model=rerank_model,
        )

    except Exception as e:
        yield f"data: Error during streaming: {str(e)}\n\n"


def build_context_string(rag_results: dict) -> str:
    """
    Convert the dictionary of RAG results into a text block
    that includes relevant metadata and passages.
    """
    lines = []
    for db_type in ["chroma", "bm25"]:
        for doc in rag_results[db_type]:
            # Access doc.base_document.text, doc.base_document.doc_id, doc.metadata, etc.
            # lines.append(f"DocID: {doc.base_document.doc_id}")
            lines.append(f"File: {doc.base_document.metadata['file_path']}")
            lines.append(f"Page: {doc.base_document.metadata['page_number']}")
            lines.append(f"Text: {doc.enhanced_text}")
            lines.append("---")

    return "\n".join(lines)


@app.post("/users/{user_id}/apps/{app_name}/sessions/{session_id}/queries")
async def handle_query(user_id: str, app_name: str, session_id: str, request: Request):
    # (Same as before)
    try:
        data = await request.json()
        user_msg_id = data.get("user_msg_id", "")
        system_msg_id = data.get("system_msg_id", "")
        assistant_msg_id = data.get("assistant_msg_id", "")
        round_id = data.get("round_id", 0)
        messages_list = data.get("messages", [])  # an array of {role, content}
        # user_query = data.get("query", "")
    except Exception as e:
        return StreamingResponse(
            iter([f"data: Error parsing request: {str(e)}\n\n"]),
            media_type="text/event-stream",
        )

    user_query = ""
    for msg in reversed(messages_list):
        if msg["role"] == "user":
            user_query = msg["content"]
            break

    # Example usage of generate_queries_from_history, if needed:
    instructions = (
        "Your task is to reconstruct the user’s request from the conversation above so it can "
        "effectively retrieve relevant documents using both vector search and BM25+ search in **Japanese**. "
        "If needed, split the request into multiple queries. You must respond **only** with "
        "strictly valid JSON, containing an array of objects, each with a single 'query' key. "
        "No extra text or explanation is allowed. For example:\n"
        "[\n"
        '  {"query": "Example query 1"},\n'
        '  {"query": "Example query 2"}\n'
        "]\n"
        "If only one query is sufficient, you may include just one object. Make sure you "
        "accurately capture the user’s intent, adding any relevant context if necessary to "
        "produce clear, natural-sounding queries."
    )

    try:
        optimized_queries = generate_queries_from_history(messages_list, instructions)
        logger.info(f"Optimized queries from LLM: {optimized_queries}")
    except ValueError as exc:
        logger.warning(
            f"Failed to generate optimized queries. Using fallback. Error: {exc}"
        )
        optimized_queries = [user_query]

    search_results = {
        "chroma": [],
        "bm25": [],
    }

    for query in optimized_queries:
        top_k_for_chroma = 20 // (len(optimized_queries))
        top_k_for_bm25 = 8 // (len(optimized_queries))
        enhance_num_brefore = 2
        enhance_num_after = 3

        if chroma_repo:
            embedded_query = embedder(query)
            logger.info(f"Searching in ChromaDB for query: {query}")
            partial_chroma = chroma_repo.search(
                query, top_k=top_k_for_chroma, query_embeddings=embedded_query
            )
            search_results["chroma"].extend(partial_chroma)

        if bm25_repo:
            logger.info(f"Searching in BM25 for query: {query}")
            partial_bm25 = bm25_repo.search(query, top_k=top_k_for_bm25)
            search_results["bm25"].extend(partial_bm25)

    enhanced_results = {}
    enhanced_results["chroma"] = chroma_repo.enhance(
        search_results["chroma"],
        num_before=enhance_num_brefore,
        num_after=enhance_num_after,
    )
    enhanced_results["bm25"] = chroma_repo.enhance(
        search_results["bm25"],
        num_before=enhance_num_brefore,
        num_after=enhance_num_after,
    )

    # please write rerank process

    retrieved_contexts_str = build_context_string(enhanced_results)

    return StreamingResponse(
        stream_chat_completion(
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
            round_id=round_id,
            user_msg_id=user_msg_id,
            system_msg_id=system_msg_id,
            assistant_msg_id=assistant_msg_id,
            messages=messages_list,
            retrieved_contexts_str=retrieved_contexts_str,
        ),
        media_type="text/event-stream",
    )
