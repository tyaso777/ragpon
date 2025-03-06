# %%
from typing import Optional, Type, TypeVar

import pandas as pd
from IPython.display import display
from peewee import SqliteDatabase

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document
from ragpon.repository.abstract_repository import AbstractRepository
from ragpon.repository.bm25.bm25_calculators import BM25PlusCalculator
from ragpon.repository.bm25.managers import DocumentManager, IndexManager
from ragpon.repository.bm25.models import DataModels
from ragpon.tokenizer import AbstractTokenizer, SudachiTokenizer

# Initialize logger
logger = get_library_logger(__name__)


# Type Aliases
TMetadata = TypeVar("TMetadata", bound=BaseDocument)
TResult = TypeVar("TResult", bound=Document)


# %%
class BM25Repository(AbstractRepository[TMetadata, TResult]):
    """
    A repository class for managing documents and performing BM25-based operations.
    """

    def __init__(
        self,
        schema: Type[TMetadata],
        result_class: Type[TResult],
        tokenizer: AbstractTokenizer = SudachiTokenizer(),
        db_path: Optional[str] = None,
        db: Optional[SqliteDatabase] = None,
    ) -> None:
        """
        Initializes the BM25Repository.

        Args:
            schema (Type[TMetadata]): Schema class for the document structure.
            result_class (Type[TResult]): The result document class.
            tokenizer (AbstractTokenizer): The tokenizer instance.
            db (Optional[SqliteDatabase]): An existing SQLite connection.
            db_path (Optional[str]): Path to the SQLite database if db is not provided.
        """
        logger.info("Initializing BM25Repository.")

        if db is not None:
            self._db = db
        else:
            self._db_path = db_path
            self._db = self._connect_to_database()

        self._result_class = result_class
        self._models = DataModels(db=self._db, schema=schema)
        self._tokenizer = tokenizer
        self._index_manager = IndexManager(data_models=self._models)
        self._doc_manager = DocumentManager(
            db=self._db,
            index_manager=self._index_manager,
            tokenizer=self._tokenizer,
            data_models=self._models,
        )

        self._bm_calculator = BM25PlusCalculator(
            data_models=self._models,
            tokenizer=self._tokenizer,
        )
        logger.info("BM25Repository initialized successfully.")

    def _connect_to_database(self) -> SqliteDatabase:
        """
        Connects to the SQLite database.

        Returns:
            SqliteDatabase: The database connection.
        """
        logger.info("Connecting to database.")
        db = SqliteDatabase(
            (
                ":memory:"
                if self._db_path is None or self._db_path == ":memory:"
                else self._db_path
            ),
            pragmas={
                "journal_mode": "wal",  # Enable concurrent read access
                "cache_size": -200000,  # Increase cache size for performance
                "synchronous": "normal",  # Optimize write performance
                "temp_store": "memory",  # Store temporary tables in memory for faster access
            },
            check_same_thread=False,  # **Allow multi-threaded access**
        )
        db.connect()
        logger.info("Database connected successfully.")
        return db

    def insert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        """
        Inserts new documents into the repository.

        Args:
            documents (list[str]): List of document contents.
            metadatas (list[TMetadata]): List of metadata associated with each document.
        """
        logger.info("Inserting %d documents.", len(documents))
        for doc, meta in zip(documents, metadatas):
            self._doc_manager.add_document(text=doc, metadata=meta)

    def upsert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        """
        Inserts or updates documents in the repository.

        Args:
            documents (list[str]): List of document contents.
            metadatas (list[TMetadata]): List of metadata associated with each document.
        """
        logger.info("Upserting %d documents.", len(documents))
        for doc, meta in zip(documents, metadatas):
            self._doc_manager.upsert_document(text=doc, metadata=meta)

    def delete_by_ids(self, ids: list[str]) -> None:
        """
        Deletes documents by their IDs.

        Args:
            ids (list[str]): List of document IDs to delete.
        """
        logger.info("Deleting documents by IDs.")
        for id in ids:
            self._doc_manager.remove_document(doc_id=id)

    def delete_by_metadata(self, metadata: dict) -> None:
        """
        Deletes documents by metadata.

        Args:
            metadata (dict): Metadata to match for deletion.
        """
        logger.info("Deleting documents by metadata: %s", metadata)
        self._doc_manager.remove_document_by_metadata(metadata=metadata)

    def search(self, query: str, top_k: int = 10) -> list[TResult]:
        """
        Performs a BM25 search in the repository.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list[TResult]: List of search results.
        """
        logger.info("Performing search with query: '%s'", query)
        results = self._bm_calculator.search(query=query, top_k=top_k)
        logger.info(
            "Search completed successfully. Retrieved %d results.", len(results)
        )
        return results

    def enhance(
        self, docs: TResult | list[TResult], num_before: int = 1, num_after: int = 1
    ) -> TResult | list[TResult]:
        """
        Enhance a document or a list of documents by combining context from surrounding documents.

        Args:
            docs (TResult | list[TResult]): The base document or list of documents to enhance.
            num_before (int): Number of preceding documents to include.
            num_after (int): Number of following documents to include.

        Returns:
            TResult | list[TResult]: Enhanced document(s) with additional context.
        """
        try:
            if isinstance(docs, list):
                logger.info("Enhancing a list of documents.")
                return [self._enhance_single(d, num_before, num_after) for d in docs]
            else:
                logger.info("Enhancing a single document.")
                return self._enhance_single(docs, num_before, num_after)
        except Exception as e:
            logger.error("Error enhancing document(s). Error: %s", str(e))
            raise

    def _enhance_single(self, doc: TResult, num_before: int, num_after: int) -> TResult:
        """
        Enhance a single document by combining context from surrounding documents.

        Args:
            doc (TResult): The base document to enhance.
            num_before (int): Number of preceding documents to include.
            num_after (int): Number of following documents to include.

        Returns:
            TResult: Enhanced document with additional context.
        """
        try:
            logger.info(
                "Enhancing document with ID: %s.",
                doc.base_document.doc_id,
            )
            # Extract IDs for the surrounding documents
            filename, base_serial_num = doc.base_document.doc_id.split("_No.")
            surrounding_ids = [
                f"{filename}_No.{serial_num}"
                for serial_num in range(
                    int(base_serial_num) - num_before,
                    int(base_serial_num) + num_after + 1,
                )
            ]

            # Retrieve surrounding documents from the database
            surrounding_docs = self._get_surrounding_documents(surrounding_ids)

            # Combine text from surrounding documents
            combined_text = "".join([d.text for d in surrounding_docs if d])

            # Create an enhanced document
            enhanced_doc = self._result_class(
                base_document=doc.base_document,
                enhanced_text=combined_text,
            )
            logger.info(
                "Document with ID: %s enhanced_text successfully.",
                doc.base_document.doc_id,
            )
            return enhanced_doc
        except Exception as e:
            logger.error(
                "Error enhancing document with ID: %s. Error: %s",
                doc.base_document.doc_id,
                str(e),
            )
            raise

    def _get_surrounding_documents(self, ids: list[str]) -> list[TResult]:
        """
        Retrieve documents by their IDs.

        Args:
            ids (list[str]): List of document IDs to retrieve.

        Returns:
            list[TResult]: List of retrieved documents.
        """
        surrounding_docs = []
        for doc_id in ids:
            try:
                doc = self._models.Document.get(self._models.Document.id == doc_id)
                surrounding_docs.append(doc)
            except self._models.Document.DoesNotExist:
                logger.warning("Document with ID %s not found.", doc_id)
                continue
        return surrounding_docs

    def _debug_print_all_records(self):
        """
        Prints all records from the database for debugging purposes.
        """
        logger.info("Printing all records for debugging.")

        model_attributes = ["Statistics", "Document", "Term", "TermFrequency"]
        for attr_name in model_attributes:
            model = getattr(self._models, attr_name)
            query = model.select()
            df = pd.DataFrame(list(query.dicts()))
            print(f"Records from {model.__name__}:")
            display(df)

    # for ContextManager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.close()

    def __del__(self):
        """
        Ensures the database connection is closed when the instance is deleted.
        """
        try:
            self._db.close()
        except Exception as e:
            logger.error("Failed to close the database connection: %s", str(e))


# %%
