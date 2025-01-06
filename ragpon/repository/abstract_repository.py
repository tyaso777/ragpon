from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document

# Initialize logger
logger = get_library_logger(__name__)

# Type variables for metadata and result types
TMetadata = TypeVar("TMetadata", bound=BaseDocument)
TResult = TypeVar("TResult", bound=Document)


class AbstractRepository(ABC, Generic[TMetadata, TResult]):
    """
    Abstract base class for repositories.

    This class defines the common interface for interacting with different types of
    repositories, such as inserting, updating, deleting, searching, and enhancing
    documents in a storage backend.

    Generic parameters:
        TMetadata: The type of metadata associated with documents.
        TResult: The type of documents stored and retrieved by the repository.
    """

    def __init__(
        self,
        metadata_class: Type[TMetadata],
        result_class: Type[TResult],
    ):
        """
        Initialize the repository with metadata and result classes.

        Args:
            metadata_class (Type[TMetadata]): Class representing the metadata structure.
            result_class (Type[TResult]): Class representing the document structure.
        """
        self.metadata_class = metadata_class
        self.result_class = result_class

    @abstractmethod
    def insert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        """
        Insert a batch of documents with their associated metadata into the repository.

        Args:
            documents (list[str]): List of document contents as strings.
            metadatas (list[TMetadata]): List of metadata associated with each document.
        """
        pass

    @abstractmethod
    def upsert(self, documents: list[str], metadatas: list[TMetadata]) -> None:
        """
        Insert or update a batch of documents with their associated metadata.

        If a document already exists, its metadata will be updated.

        Args:
            documents (list[str]): List of document contents as strings.
            metadatas (list[TMetadata]): List of metadata associated with each document.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """
        Delete documents from the repository by their unique IDs.

        Args:
            ids (list[str]): List of document IDs to be deleted.
        """
        pass

    @abstractmethod
    def delete_by_metadata(self, metadata: dict) -> None:
        """
        Delete documents from the repository based on metadata filters.

        Args:
            metadata (dict): Dictionary of metadata fields and values to match.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[TResult]:
        """
        Search for documents in the repository based on a query string.

        Args:
            query (str): Search query string.
            top_k (int): Number of top results to retrieve.

        Returns:
            list[TResult]: List of documents matching the query.
        """
        pass

    @abstractmethod
    def enhance(
        self, docs: TResult | list[TResult], num_before: int = 1, num_after: int = 1
    ) -> TResult | list[TResult]:
        """
        Enhance a document or a list of documents by combining context from surrounding documents.

        Args:
            docs (Union[TResult, List[TResult]]): The base document or list of documents to enhance.
            num_before (int): Number of preceding documents to include.
            num_after (int): Number of following documents to include.

        Returns:
            Union[TResult, List[TResult]]: Enhanced document(s) with additional context.
        """
        pass
