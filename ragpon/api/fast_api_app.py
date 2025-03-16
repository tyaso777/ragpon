# %%
# FastAPI side
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
    print("[MOCK INSERT] Storing 3 records in the DB with the following data:")
    print(
        f"  user_id={user_id}, session_id={session_id}, app_name={app_name}, round_id={round_id}"
    )
    print(
        f"  user_msg_id={user_msg_id}, system_msg_id={system_msg_id}, assistant_msg_id={assistant_msg_id}"
    )
    print(f"  user_query={user_query}")
    print(f"  retrieved_contexts_str={retrieved_contexts_str}")
    print(f"  total_stream_content={total_stream_content}")
    print(f"  llm_model={llm_model}, rerank_model={rerank_model}")

    # Real code example:
    # with db_session() as sess:
    #     sess.execute("""
    #       INSERT INTO messages (id, user_id, session_id, app_name, round_id, content, type, llm_model, rerank_model)
    #       VALUES (:user_msg_id, :user_id, :session_id, :app_name, :round_id, :user_query, 'user', :llm_model, :rerank_model)
    #     """, {...})
    #
    #     sess.execute("""
    #       INSERT INTO messages (id, user_id, session_id, app_name, round_id, content, type, llm_model, rerank_model)
    #       VALUES (:assistant_msg_id, :user_id, :session_id, :app_name, :round_id, :total_stream_content, 'assistant', :llm_model, :rerank_model)
    #     """, {...})
    #
    #     # If you have a system message too:
    #     # sess.execute(...)
    #
    #     sess.commit()


def stream_chat_completion(
    user_id: str,
    session_id: str,
    app_name: str,
    round_id: int,
    user_msg_id: str,
    system_msg_id: str,
    assistant_msg_id: str,
    user_query: str,
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
        user_query (str): The user's query text to be processed by the LLM.
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
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. In your responses, please "
                        "include the file names and page numbers that serve as "
                        "the basis for your statements..."
                    ),
                },
                {"role": "system", "content": f"Context: {retrieved_contexts_str}"},
                {"role": "user", "content": user_query},
            ],
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
        user_query = data.get("query", "")
    except Exception as e:
        return StreamingResponse(
            iter([f"data: Error parsing request: {str(e)}\n\n"]),
            media_type="text/event-stream",
        )

    top_k_for_chroma = 20
    top_k_for_bm25 = 5
    enhance_num_brefore = 2
    enhance_num_after = 3
    search_results: dict[str, list[Document]] = {}
    if chroma_repo:
        embedded_query = embedder(user_query)
        logger.info(f"Searching in ChromaDB for query: {user_query}")
        search_results["chroma"] = chroma_repo.search(
            user_query, top_k=top_k_for_chroma, query_embeddings=embedded_query
        )
    if bm25_repo:
        logger.info(f"Searching in BM25 for query: {user_query}")
        search_results["bm25"] = bm25_repo.search(user_query, top_k=top_k_for_bm25)

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
            user_query=user_query,
            retrieved_contexts_str=retrieved_contexts_str,
        ),
        media_type="text/event-stream",
    )
