# %%
from collections import OrderedDict, defaultdict
from typing import Any

import pandas as pd

from ragpon._utils.logging_helper import get_library_logger
from ragpon.chunk_processor import AbstractChunkProcessor, JAGinzaChunkProcessor
from ragpon.config import Config
from ragpon.domain.document_processing_pipeline import (
    DataFrameDocumentProcessingPipeline,
    FilePathDocumentProcessingPipeline,
)
from ragpon.domain.document_reader import ExtensionBasedDocumentReaderFactory
from ragpon.domain.domain import BaseDocument, Document
from ragpon.domain.metadata_generator import CustomMetadataGenerator
from ragpon.ml_models.embedding_model import (
    AbstractEmbeddingModel,
    ChromaDBEmbeddingAdapter,
    MultilingualE5LargeEmbedder,
)
from ragpon.ml_models.reranker import (
    AbstractRelevanceEvaluator,
    JapaneseRerankerCrossEncoderLargeV1Evaluator,
    Reranker,
)
from ragpon.repository.bm25.bm25_repository import BM25Repository
from ragpon.repository.chromaDB_repository import ChromaDBRepository
from ragpon.tokenizer import AbstractTokenizer, SudachiTokenizer

# Initialize logger
logger = get_library_logger(__name__)


class DatabaseFactory:
    """Factory class to create BM25 or ChromaDB repositories based on the configuration."""

    def __init__(
        self,
        config: Config,
        tokenizer: AbstractTokenizer | None = None,
        embedder: AbstractEmbeddingModel | None = None,
    ) -> None:
        """Initializes the DatabaseFactory with the given configuration.

        Args:
            config (Config): The configuration object.
            tokenizer (AbstractTokenizer | None): Custom tokenizer for BM25.
            embedder (AbstractEmbeddingModel | None): Custom embedder for ChromaDB.
        """
        self._config = config
        self._tokenizer = tokenizer or SudachiTokenizer()
        self._embedder = embedder or MultilingualE5LargeEmbedder(config=self._config)

    def create_bm25(self) -> BM25Repository | None:
        """Creates a BM25Repository instance if BM25 is enabled in the configuration.

        Returns:
            BM25Repository | None: A BM25Repository instance if enabled, otherwise None.
        """
        use_bm25 = self._config.get("DATABASES.USE_BM25", "False").lower() == "true"
        if not use_bm25:
            logger.info("BM25 is disabled in the configuration.")
            return None

        db_path = self._config.get("DATABASES.BM25_PATH", None)
        if db_path == "None":  # Allow explicit "None" as string for in-memory DB
            db_path = None
        logger.info(
            f"Creating BM25 database. {'Using in-memory mode' if db_path is None else f'Path: {db_path}'}"
        )
        return BM25Repository(
            db_path=db_path,
            schema=BaseDocument,
            result_class=Document,
            tokenizer=self._tokenizer,
        )

    def create_chroma_db(self) -> ChromaDBRepository | None:
        """Creates a ChromaDBRepository instance if ChromaDB is enabled in the configuration.

        Returns:
            ChromaDBRepository | None: A ChromaDBRepository instance if enabled, otherwise None.
        """
        use_chromadb = (
            self._config.get("DATABASES.USE_CHROMADB", "False").lower() == "true"
        )
        if not use_chromadb:
            logger.info("ChromaDB is disabled in the configuration.")
            return None

        collection_name = (
            self._config.get("DATABASES.CHROMADB_COLLECTION_NAME", "default_collection")
            or "default_collection"
        )

        folder_path = self._config.get("DATABASES.CHROMADB_FOLDER_PATH", None)
        if folder_path == "None":  # Allow explicit "None" as string for in-memory DB
            folder_path = None
        logger.info(
            f"Creating ChromaDB collection '{collection_name}'. "
            f"{'Using in-memory mode' if folder_path is None else f'Folder Path: {folder_path}'}"
        )
        return ChromaDBRepository(
            collection_name=collection_name,
            embed_func=ChromaDBEmbeddingAdapter(self._embedder),
            metadata_class=BaseDocument,
            result_class=Document,
            similarity="cosine",
            folder_path=folder_path,
            batch_size=100,
        )

    def validate_configuration(self) -> None:
        """Validates that at least one database (BM25 or ChromaDB) is enabled."""
        use_bm25 = self._config.get("DATABASES.USE_BM25", "False").lower() == "true"
        use_chromadb = (
            self._config.get("DATABASES.USE_CHROMADB", "False").lower() == "true"
        )

        if not use_bm25 and not use_chromadb:
            logger.warning(
                "Both BM25 and ChromaDB are disabled. No databases will be available for operations."
            )


class DocumentProcessingService:
    """Service class to manage document processing, including insertion, search, and deletion."""

    def __init__(
        self,
        config_or_config_path: str | Config,
        reader_factory: ExtensionBasedDocumentReaderFactory | None = None,
        chunk_processor: AbstractChunkProcessor | None = None,
        tokenizer: AbstractTokenizer | None = None,
        embedder: AbstractEmbeddingModel | None = None,
        relevance_evaluator: AbstractRelevanceEvaluator | None = None,
    ) -> None:
        """Initializes the DocumentProcessingService with a given configuration path.

        Args:
            config_or_config_path (str | Config): Either the configuration file path or a Config instance.
            reader_factory (ExtensionBasedDocumentReaderFactory | None): Custom document reader factory.
            chunk_processor (AbstractChunkProcessor | None): Custom chunk processor.
            tokenizer (AbstractTokenizer | None): Custom tokenizer.
            embedder (AbstractEmbeddingModel | None): Custom embedding model.
            relevance_evaluator (AbstractRelevanceEvaluator | None): Custom relevance evaluator.
        """
        # Allow both Config instance and path to be passed
        if isinstance(config_or_config_path, Config):
            self._config = config_or_config_path
        elif isinstance(config_or_config_path, str):
            self._config = Config(config_or_config_path)
        else:
            error_message = (
                "Invalid type for config_or_config_path. Expected str or Config, "
                f"but got {type(config_or_config_path).__name__}."
            )
            logger.error(error_message)
            raise TypeError(error_message)
        self._factory = DatabaseFactory(
            config=self._config,
            tokenizer=tokenizer or SudachiTokenizer(),
            embedder=embedder or MultilingualE5LargeEmbedder(config=self._config),
        )

        # Initialize databases based on configuration
        self._bm25_repo = self._factory.create_bm25()
        self._chromadb_repo = self._factory.create_chroma_db()

        # Validate configuration to ensure at least one database is active
        self._factory.validate_configuration()

        self._metadata_generator = CustomMetadataGenerator(metadata_class=BaseDocument)
        self._chunk_processor = chunk_processor or JAGinzaChunkProcessor(chunk_size=100)
        self._reader_factory = reader_factory or ExtensionBasedDocumentReaderFactory()
        self._relevance_evaluator = (
            relevance_evaluator
            or JapaneseRerankerCrossEncoderLargeV1Evaluator(config=self._config)
        )
        logger.info("DocumentProcessingService initialized.")

    def process_file(self, file_path: str) -> None:
        """Reads a file, splits it into chunks, generates metadata,
        and saves them to the active repositories.

        Args:
            file_path (str): The path to the file.
        """
        logger.info(f"Processing File: {file_path}")
        reader = self._reader_factory.get_document_reader(file_path)
        pages = reader.read_document(start_page_number=1)
        pipeline = FilePathDocumentProcessingPipeline(
            chunk_processor=self._chunk_processor,
            metadata_generator=self._metadata_generator,
        )
        all_chunks, all_metadata = pipeline.process_document(
            pages=pages, file_path=file_path
        )
        self._save_to_repositories(all_chunks, all_metadata)

    def process_dataframe(
        self, df: pd.DataFrame, chunk_col_name: str, id_col_name: str
    ) -> None:
        """Processes a pandas DataFrame by splitting its text column into chunks
        and saving them to the repositories.

        Args:
            df (pd.DataFrame): The input dataframe containing the text data.
            chunk_col_name (str): The name of the column containing text to be chunked.
            id_col_name (str): The name of the column containing IDs for each row.
        """
        logger.info("Processing DataFrame.")
        pipeline = DataFrameDocumentProcessingPipeline(
            metadata_generator=self._metadata_generator,
            chunk_col_name=chunk_col_name,
            id_col_name=id_col_name,
        )
        all_chunks, all_metadata = pipeline.process_document(df=df)
        self._save_to_repositories(all_chunks, all_metadata)

    def _save_to_repositories(
        self, chunks: list[str], metadata: list[BaseDocument]
    ) -> None:
        """Saves data to the active repositories (BM25, ChromaDB).

        Args:
            chunks (list[str]): A list of text chunks to be stored.
            metadata (list[BaseDocument]): A list of BaseDocument objects containing metadata.
        """
        if self._chromadb_repo:
            logger.info("Saving data to ChromaDB repository.")
            self._chromadb_repo.upsert(documents=chunks, metadatas=metadata)
        if self._bm25_repo:
            logger.info("Saving data to BM25 repository.")
            self._bm25_repo.upsert(documents=chunks, metadatas=metadata)

    def delete_by_ids(self, ids: list[str]) -> None:
        """Deletes documents based on a list of IDs in both BM25 and ChromaDB if available.

        Args:
            ids (list[str]): The list of document IDs to be deleted.
        """
        logger.info(f"Deleting documents by ids: {ids}")
        if self._bm25_repo:
            logger.info("Deleting from BM25 repository.")
            self._bm25_repo.delete_by_ids(ids)
        if self._chromadb_repo:
            logger.info("Deleting from ChromaDB repository.")
            self._chromadb_repo.delete_by_ids(ids)

    def delete_by_metadata(self, metadata: dict[str, Any]) -> None:
        """Deletes documents that match the given metadata in both BM25 and ChromaDB if available.

        Args:
            metadata (dict[str, Any]): The metadata key-value pairs used for matching documents to delete.
        """
        logger.info(f"Deleting documents by metadata: {metadata}")
        if self._bm25_repo:
            logger.info("Deleting from BM25 repository.")
            self._bm25_repo.delete_by_metadata(metadata)
        if self._chromadb_repo:
            logger.info("Deleting from ChromaDB repository.")
            self._chromadb_repo.delete_by_metadata(metadata)

    def search(self, query: str, top_k: int = 10) -> dict[str, list[Document]]:
        """Searches for the top_k most relevant documents in both BM25 and ChromaDB if available.

        Args:
            query (str): The query string to search for.
            top_k (int, optional): The number of documents to retrieve. Defaults to 10.

        Returns:
            dict[str, list[Document]]: A dictionary containing search results for each active repository.
                Example:
                {
                    "chroma": [Document(...), ...],
                    "bm25": [Document(...), ...]
                }
        """
        results: dict[str, list[Document]] = {}
        if self._chromadb_repo:
            logger.info(f"Searching in ChromaDB for query: {query}")
            results["chroma"] = self._chromadb_repo.search(query, top_k=top_k)
        if self._bm25_repo:
            logger.info(f"Searching in BM25 for query: {query}")
            results["bm25"] = self._bm25_repo.search(query, top_k=top_k)
        if not results:
            logger.warning("No active repositories available for search.")
        return results

    def enhance_search_results(
        self,
        search_results: dict[str, list[Document]],
        num_before: int = 1,
        num_after: int = 1,
    ) -> dict[str, list[Document]]:
        """Enhances search results by adding contextual chunks (before and after)
        in the active repositories.

        If ChromaDB is active, it will be used. Otherwise, BM25 will be used.

        Args:
            search_results (dict[str, list[Document]]): The search results grouped by repository.
            num_before (int, optional): Number of chunks to retrieve before the matched document. Defaults to 1.
            num_after (int, optional): Number of chunks to retrieve after the matched document. Defaults to 1.

        Returns:
            list[Document]: The enhanced list of documents with additional context.
        """
        logger.info("Enhancing search results with context.")
        enhanced_results: dict[str, list[Document]] = {}

        if "chroma" in search_results and self._chromadb_repo:
            logger.info("Using ChromaDB for enhancing search results.")
            enhanced_results["chroma"] = self._chromadb_repo.enhance(
                search_results["chroma"], num_before, num_after
            )

        if "bm25" in search_results and self._bm25_repo:
            logger.info("Using BM25 for enhancing search results.")
            enhanced_results["bm25"] = self._bm25_repo.enhance(
                search_results["bm25"], num_before, num_after
            )

        if not enhanced_results:
            logger.warning(
                "No active repositories available for enhancing search results."
            )

        return enhanced_results

    def rerank_results(
        self,
        query: str,
        search_results: dict[str, list[Document]],
        search_result_text_key: str = "enhanced_text",
    ) -> dict[str, list[Document]]:
        """Reranks the given search results using a cross-encoder model.

        Args:
            query (str): The query string to evaluate relevance against.
            search_results (dict[str, list[Document]]): The search results grouped by repository.
            search_result_text_key (str, optional): The string in Document to evaluate relevance against. Defaults to "enhanced_text".

        Returns:
            dict[str, list[tuple[Document, float]]]: A dictionary with reranked results and their scores.
        """
        logger.info("Reranking results.")
        reranked_results: dict[str, list[Document]] = {}
        reranker = Reranker(relevance_evaluator=self._relevance_evaluator)

        if "chroma" in search_results:
            logger.info("Reranking ChromaDB results.")
            reranked_results["chroma"] = reranker.rerank(
                query=query,
                search_results=search_results["chroma"],
                search_result_text_key=search_result_text_key,
            )

        if "bm25" in search_results:
            logger.info("Reranking BM25 results.")
            reranked_results["bm25"] = reranker.rerank(
                query=query,
                search_results=search_results["bm25"],
                search_result_text_key=search_result_text_key,
            )

        return reranked_results

    def rerank_results_by_score_fusion(
        self, search_results: dict[str, list[Document]], alpha: float = 0.7
    ) -> dict[str, list[Document]]:
        """Performs ranking by score fusion between BM25+ and ChromaDB results.

        Args:
            search_results (dict[str, list[Document]]): A dictionary containing lists of Document objects.
            alpha (float, optional): Weight for BM25+ score (default: 0.7).

        Returns:
            dict[str, list[Document]]: A dictionary with reranked results in descending order.
        """
        logger.info("Reranking results using score fusion (BM25+ & ChromaDB).")

        # Copy search results to avoid modifying the original data
        reranked_search_results = {
            "bm25": [
                Document(
                    base_document=doc.base_document,
                    enhanced_text=doc.enhanced_text,
                    rerank=0.0,
                )
                for doc in search_results.get("bm25", [])
            ],
            "chroma": [
                Document(
                    base_document=doc.base_document,
                    enhanced_text=doc.enhanced_text,
                    rerank=0.0,
                )
                for doc in search_results.get("chroma", [])
            ],
        }

        # Create a dictionary to store distances {doc_id: [bm25_distance, chroma_distance]}
        distance_dict = {}

        for doc in reranked_search_results["bm25"]:
            doc_id = doc.base_document.doc_id
            distance_dict.setdefault(doc_id, [None, None])[0] = (
                1 - doc.base_document.distance
            )  # Convert to score

        for doc in reranked_search_results["chroma"]:
            doc_id = doc.base_document.doc_id
            distance_dict.setdefault(doc_id, [None, None])[1] = (
                1 - doc.base_document.distance
            )  # Convert to score

        # Compute fused score and assign to documents
        for doc in reranked_search_results["bm25"]:
            doc_id = doc.base_document.doc_id
            bm25_score, chroma_score = distance_dict[doc_id]
            doc.rerank = (
                alpha * bm25_score + (1 - alpha) * chroma_score
                if chroma_score is not None
                else bm25_score
            )

        for doc in reranked_search_results["chroma"]:
            doc_id = doc.base_document.doc_id
            bm25_score, chroma_score = distance_dict[doc_id]
            doc.rerank = (
                alpha * bm25_score + (1 - alpha) * chroma_score
                if bm25_score is not None
                else chroma_score
            )

        # Sort results in descending order (higher score is better)
        reranked_search_results["bm25"] = sorted(
            reranked_search_results["bm25"], key=lambda x: x.rerank, reverse=True
        )
        reranked_search_results["chroma"] = sorted(
            reranked_search_results["chroma"], key=lambda x: x.rerank, reverse=True
        )

        return reranked_search_results

    def convert_search_results_to_dataframe(
        self, search_results: dict[str, list[Document]]
    ) -> pd.DataFrame:
        """Converts search results into a flat DataFrame, merging BM25 and ChromaDB results for the same doc_id.

        This method extracts all relevant fields from the search results while preserving the order of metadata fields as they appear.
        The 'enhanced_text' and 'rerank_score' columns are placed at the end.

        Args:
            search_results (dict[str, list[Document]]): A dictionary containing lists of Document objects.

        Returns:
            pd.DataFrame: A structured DataFrame where each row represents a unique document with merged scores.
        """
        logger.info(
            "Converting search results to a DataFrame with ordered metadata fields."
        )

        # Dictionary to store merged document data using doc_id as key
        doc_dict = defaultdict(
            lambda: OrderedDict(
                {
                    "source": set(),  # Stores all sources (bm25, chroma) where the doc appeared
                    "doc_id": None,
                    "text": None,
                    "distance_bm25": None,
                    "distance_ChromaDB": None,
                }
            )
        )

        metadata_order = (
            OrderedDict()
        )  # Keeps track of the first appearance order of metadata keys

        # Iterate over both "bm25" and "chroma" results
        for source, documents in search_results.items():
            for doc in documents:
                doc_id = doc.base_document.doc_id
                db_name = doc.base_document.db_name  # Either "bm25" or "ChromaDB"
                distance_col = f"distance_{db_name}"  # Format distance_{db_name}

                # Ensure doc_id exists in the dictionary
                if doc_id not in doc_dict:
                    doc_dict[doc_id].update(
                        {
                            "doc_id": doc_id,
                            "text": doc.base_document.text,
                            "enhanced_text": doc.enhanced_text,  # Will be placed at the end
                            "rerank_score": doc.rerank,  # Will be placed at the end
                        }
                    )

                # Store source (bm25/chroma) and distance
                doc_dict[doc_id]["source"].add(source)
                doc_dict[doc_id][
                    distance_col
                ] = doc.base_document.distance  # Add distance_{db_name}

                # Merge metadata dynamically while tracking the order they first appeared
                for key, value in doc.base_document.metadata.items():
                    if key not in metadata_order:
                        metadata_order[key] = None  # Preserve order of first occurrence
                    if key not in doc_dict[doc_id]:
                        doc_dict[doc_id][
                            key
                        ] = value  # Add new metadata field if not already present

        # Convert dictionary to a DataFrame
        df = pd.DataFrame(doc_dict.values())

        # Convert "source" from set to comma-separated string
        df["source"] = df["source"].apply(lambda x: ", ".join(sorted(x)))

        # Define fixed column order: Keep metadata fields in the order they first appeared
        fixed_columns = [
            "source",
            "doc_id",
            "text",
            "distance_bm25",
            "distance_ChromaDB",
        ]
        metadata_columns = [
            col for col in metadata_order.keys() if col in df.columns
        ]  # Preserve first-seen order
        final_columns = (
            fixed_columns + metadata_columns + ["enhanced_text", "rerank_score"]
        )  # Move enhanced_text and rerank_score to the end

        # Reorder DataFrame columns
        df = df.reindex(
            columns=[col for col in final_columns if col in df.columns], fill_value=None
        )

        logger.info(
            f"Converted search results into a DataFrame with {len(df)} unique records."
        )

        return df
