from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from chromadb import QueryResult

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document

# Initialize logger
logger = get_library_logger(__name__)

T = TypeVar("T")
TResult = TypeVar("TResult", bound=Document)
TMetadata = TypeVar("TMetadata", bound=BaseDocument)


class AbstractResultsFormatter(ABC, Generic[T, TResult]):
    def __init__(self, result_class: Type[TResult]):
        self.result_class = result_class

    @abstractmethod
    def format(self, results: T) -> list[TResult]:
        pass


class ChromaDBResultsFormatter(AbstractResultsFormatter[QueryResult, TResult]):
    def __init__(self, result_class: Type[TResult]):
        super().__init__(result_class)
        logger.info(
            "ChromaDBResultsFormatter initialized with result class: %s",
            result_class.__name__,
        )

    def format(self, results: QueryResult) -> list[TResult]:
        search_results = []
        db_name = "ChromaDB"
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        logger.info("Formatting results from ChromaDB.")
        for id, distance, metadata, text in zip(ids, distances, metadatas, documents):
            try:
                # Create the BaseDocument from metadata and text
                base_document = BaseDocument(
                    doc_id=id,
                    text=text,
                    db_name=db_name,
                    distance=distance,
                    metadata=metadata,
                )

                # Initialize the Document with the base_document
                result = self.result_class(base_document=base_document)
                search_results.append(result)
            except Exception as e:
                logger.error(
                    "Error formatting result with id: %s. Error: %s", id, str(e)
                )
                raise
        logger.info(
            "Formatting completed successfully. %d results formatted.",
            len(search_results),
        )
        return search_results
