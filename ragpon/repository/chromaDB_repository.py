from typing import Any, Callable, Optional, Type, TypeVar

import chromadb
from chromadb import Documents, Embeddings, GetResult

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document
from ragpon.repository.abstract_repository import AbstractRepository
from ragpon.repository.search_results_formatter import ChromaDBResultsFormatter
from ragpon.ml_models.embedding_model import ChromaDBEmbeddingAdapter

# Initialize logger
logger = get_library_logger(__name__)

TMetadata = TypeVar("TMetadata", bound=BaseDocument)
TResult = TypeVar("TResult", bound=Document)


class ChromaDBRepository(AbstractRepository[TMetadata, TResult]):
    def __init__(
        self,
        collection_name: str,
        embed_func: ChromaDBEmbeddingAdapter,
        # embed_func: Callable[[Documents], Embeddings],
        metadata_class: Type[TMetadata],
        result_class: Type[TResult],
        similarity: str = "cosine",
        folder_path: Optional[str] = None,
        batch_size: int = 100,
    ):
        logger.info("Initializing ChromaDB for collection: %s", collection_name)
        self.collection_name = collection_name
        self.embed_func = embed_func
        self.query_prefix = embed_func.query_prefix
        self.passage_prefix = embed_func.passage_prefix
        self.similarity = similarity
        self.batch_size = batch_size
        self.metadata_class = metadata_class
        self.result_class = result_class
        self.formatter = ChromaDBResultsFormatter(result_class=result_class)
        try:
            if folder_path:
                self.client = chromadb.PersistentClient(folder_path)
                logger.info("Using persistent storage at: %s", folder_path)
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embed_func,
                metadata={"hnsw:space": self.similarity},
            )
            logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            logger.error("Error initializing ChromaDB: %s", str(e))
            raise

    def upsert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        try:
            logger.info(
                "Upserting %d documents into collection: %s",
                len(documents),
                self.collection_name,
            )
            for metadata in metadatas:
                assert hasattr(
                    metadata, "doc_id"
                ), "metadata should include 'doc_id', but it's missing."
            ids = [metadata.doc_id for metadata in metadatas]
            other_metadatas = [
                {k: v for k, v in vars(metadata).items() if k != "doc_id"}["metadata"]
                for metadata in metadatas
            ]

            doc_batches = self._batch_process(documents, self.batch_size)
            metadata_batches = self._batch_process(other_metadatas, self.batch_size)
            id_batches = self._batch_process(ids, self.batch_size)

            for data_batch, metadata_batch, ids_batch in zip(
                doc_batches, metadata_batches, id_batches
            ):

                data_batch = [f"{self.passage_prefix}{doc}" for doc in data_batch]

                self.collection.upsert(
                    documents=data_batch,
                    metadatas=metadata_batch,
                    ids=ids_batch,
                )
            logger.info("Upsert completed successfully.")
        except Exception as e:
            logger.error("Error during upsert: %s", str(e))
            raise

    def insert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        try:
            logger.info(
                "Inserting %d documents into collection: %s",
                len(documents),
                self.collection_name,
            )
            for metadata in metadatas:
                assert hasattr(
                    metadata, "doc_id"
                ), "metadata should include 'doc_id', but it's missing."
            ids = [metadata.doc_id for metadata in metadatas]
            other_metadatas = [
                {k: v for k, v in vars(metadata).items() if k != "doc_id"}["metadata"]
                for metadata in metadatas
            ]

            doc_batches = self._batch_process(documents, self.batch_size)
            metadata_batches = self._batch_process(other_metadatas, self.batch_size)
            id_batches = self._batch_process(ids, self.batch_size)

            for data_batch, metadata_batch, ids_batch in zip(
                doc_batches, metadata_batches, id_batches
            ):

                data_batch = [f"{self.passage_prefix}{doc}" for doc in data_batch]

                self.collection.add(
                    documents=data_batch,
                    metadatas=metadata_batch,
                    ids=ids_batch,
                )
            logger.info("Insert completed successfully.")
        except Exception as e:
            logger.error("Error during insert: %s", str(e))
            raise

    def _batch_process(self, data: list[Any], batch_size: int):
        logger.debug("Batch processing data with batch size: %d", batch_size)
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def delete_by_ids(self, ids: list[str]) -> None:
        try:
            logger.info(
                "Deleting documents by IDs in collection: %s", self.collection_name
            )
            self.collection.delete(ids=ids)
            logger.info("Delete by IDs completed successfully.")
        except Exception as e:
            logger.error("Error during delete by IDs: %s", str(e))
            raise

    def delete_by_metadata(self, metadata: dict) -> None:
        try:
            logger.info(
                "Deleting documents by metadata in collection: %s", self.collection_name
            )
            self.collection.delete(where=metadata)
            logger.info("Delete by metadata completed successfully.")
        except Exception as e:
            logger.error("Error during delete by metadata: %s", str(e))
            raise

    def search(
        self, query: str, top_k: int = 10, where: Optional[dict] = None
    ) -> list[TResult]:
        try:
            logger.info(
                "Searching in collection: %s with query: '%s'",
                self.collection_name,
                query,
            )

            prefixed_query = f"{self.query_prefix}{query}"

            if where is None:
                search_result = self.collection.query(
                    query_texts=prefixed_query,
                    n_results=top_k,
                )
            else:
                search_result = self.collection.query(
                    query_texts=prefixed_query,
                    n_results=top_k,
                    where=where,
                )
            results = self.formatter.format(search_result)
            logger.info(
                "Search completed successfully. Retrieved %d results.", len(results)
            )
            return results
        except Exception as e:
            logger.error("Error during search: %s", str(e))
            raise

    def enhance(
        self, docs: TResult | list[TResult], num_before: int = 1, num_after: int = 1
    ) -> TResult | list[TResult]:
        """
        Enhance a document or a list of documents by combining context from surrounding documents.

        Args:
            doc (Union[TResult, List[TResult]]): The base document or list of documents to enhance.
            num_before (int): Number of preceding documents to include.
            num_after (int): Number of following documents to include.

        Returns:
            Union[TResult, List[TResult]]: Enhanced document(s) with additional context.
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
                "Enhancing document with ID: %s in collection: %s",
                doc.base_document.doc_id,
                self.collection_name,
            )
            # Extract IDs for the surrounding documents
            filename, base_serial_num = doc.base_document.doc_id.split("_No.")
            ids = [
                f"{filename}_No.{serial_num}"
                for serial_num in range(
                    int(base_serial_num) - num_before,
                    int(base_serial_num) + num_after + 1,
                )
            ]

            # Retrieve surrounding documents
            surrounding_docs = self._get_by_id(ids)

            # Combine text from surrounding documents
            combined_text = "".join(
                [txt.removeprefix(self.passage_prefix) for txt in surrounding_docs["documents"]]
            )

            # Create an enhanced document
            enhanced_doc = self.result_class(
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

    def _get_by_id(self, ids: list[str], where: Optional[dict] = None) -> GetResult:
        """
        Retrieve documents by their IDs from ChromaDB.

        Args:
            ids (list[str]): List of document IDs to retrieve.
            where (Optional[dict]): Optional filtering criteria.

        Returns:
            GetResult: Retrieved documents and metadata in ChromaDB's native format.
        """
        try:
            logger.info(
                "Retrieving documents by IDs in collection: %s", self.collection_name
            )
            if where is None:
                results = self.collection.get(ids=ids)
            else:
                results = self.collection.get(ids=ids, where=where)
            logger.info(
                "Get by IDs completed successfully. Retrieved %d documents.",
                len(results["documents"]),
            )
            return results
        except Exception as e:
            logger.error("Error during get by ID: %s", str(e))
            raise
