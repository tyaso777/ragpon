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


def stream_chat_completion(
    user_query: str, contexts: list
) -> Generator[str, None, None]:
    try:
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. In your responses, please include the file names and page numbers that serve as the basis for your statements. Make sure the user can clearly see which documents and pages support your answers. If you do not have enough information, acknowledge that instead of making assumptions.",
                },
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
        user_query = data.get("query", "")
        file_info = data.get("file", None)
        is_private = data.get("is_private_session", False)
    except Exception as e:
        return StreamingResponse(
            iter([f"data: Error parsing request: {str(e)}\n\n"]),
            media_type="text/event-stream",
        )

    top_k = 20
    enhance_num_brefore = 2
    enhance_num_after = 3
    search_results: dict[str, list[Document]] = {}
    if chroma_repo:
        embedded_query = embedder(user_query)
        logger.info(f"Searching in ChromaDB for query: {user_query}")
        search_results["chroma"] = chroma_repo.search(
            user_query, top_k=top_k, query_embeddings=embedded_query
        )
    if bm25_repo:
        logger.info(f"Searching in BM25 for query: {user_query}")
        search_results["bm25"] = bm25_repo.search(user_query, top_k=top_k)

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

    # retrieved_contexts = ["Some retrieved context 1", "Some retrieved context 2"]
    return StreamingResponse(
        stream_chat_completion(user_query, [retrieved_contexts_str]),
        media_type="text/event-stream",
    )
