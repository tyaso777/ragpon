# %%
import json
from typing import Any

from peewee import SqliteDatabase, fn

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument
from ragpon.repository.bm25.models import DataModels
from ragpon.tokenizer import AbstractTokenizer

# Initialize logger
logger = get_library_logger(__name__)


# %%
class IndexManager:
    """
    Manages term indexes and statistics in the BM25 repository.

    Attributes:
        _DataModels (DataModels): The data models instance containing the database schema.
    """

    def __init__(self, data_models: DataModels) -> None:
        """
        Initializes the IndexManager.

        Args:
            data_models (DataModels): The data models instance containing the database schema.
        """
        self._DataModels = data_models

    def update_term_index(
        self, word: str, doc_id: str, count: int, remove: bool = False
    ) -> None:
        """
        Updates the term index for a given word and document.

        Args:
            word (str): The term to update.
            doc_id (str): The document ID associated with the term.
            count (int): The term frequency in the document.
            remove (bool): Whether to remove the term from the index.
        """
        logger.info(
            "Updating term index for word '%s' in document ID '%s'.", word, doc_id
        )
        term, created = self._DataModels.Term.get_or_create(
            term=word, defaults={"document_frequency": 0}
        )
        if remove:
            term.document_frequency -= 1
            if term.document_frequency <= 0:
                term.delete_instance()
                logger.info("Term '%s' deleted from index.", word)
            else:
                term.save()
            self._DataModels.TermFrequency.delete().where(
                self._DataModels.TermFrequency.term == term,
                self._DataModels.TermFrequency.document_id == doc_id,
            ).execute()
        else:
            if created:
                logger.info("Term '%s' added to index.", word)
                term.document_frequency = 1
            else:
                term.document_frequency += 1
            term.save()
            self._DataModels.TermFrequency.create(
                term=term, document_id=doc_id, term_frequency=count
            )
        logger.info("Term index updated successfully for word '%s'.", word)

    def update_statistics(self, words: list[str], add: bool = True) -> None:
        """
        Updates document statistics.

        Args:
            words (list[str]): The words in the document.
            add (bool): Whether to add or subtract the statistics.
        """
        logger.info("Updating document statistics.")
        stats, created = self._DataModels.Statistics.get_or_create(
            id=1, defaults={"total_documents": 0, "total_words": 0}
        )
        if created:
            logger.info("Statistics record created.")
            stats.total_documents = 1
            stats.total_words = len(words)
        else:
            if add:
                stats.total_documents += 1
                stats.total_words += len(words)
            else:
                stats.total_documents -= 1
                stats.total_words -= len(words)
        stats.save()
        logger.info("Document statistics updated successfully.")


class DocumentManager:
    """
    Manages documents in the BM25 repository.

    Attributes:
        _db (SqliteDatabase): The database connection.
        _index_manager (IndexManager): The index manager instance.
        _tokenizer (AbstractTokenizer): The tokenizer instance.
        _data_models (DataModels): The data models instance containing the database schema.
    """

    def __init__(
        self,
        db: SqliteDatabase,
        index_manager: IndexManager,
        tokenizer: AbstractTokenizer,
        data_models: DataModels,
    ):
        """
        Initializes the DocumentManager.

        Args:
            db (SqliteDatabase): The database connection.
            index_manager (IndexManager): The index manager instance.
            tokenizer (AbstractTokenizer): The tokenizer instance.
            data_models (DataModels): The data models instance containing the database schema.
        """
        self._db = db
        self._index_manager = index_manager
        self._tokenizer = tokenizer
        self._data_models = data_models

    def add_document(self, text: str, metadata: BaseDocument) -> None:
        """
        Adds a new document to the repository.

        Args:
            text (str): The content of the document.
            metadata (BaseDocument): The metadata associated with the document.
        """
        logger.info("Adding document with metadata ID '%s'.", metadata.doc_id)

        with self._db.atomic():
            document_id = metadata.doc_id
            existing_document = self._data_models.Document.get_or_none(
                self._data_models.Document.id == document_id
            )

            if existing_document is not None:
                logger.warning("Document with ID '%s' already exists.", document_id)
                return

            # Extract db_name and distance with defaults
            db_name = metadata.metadata.pop("db_name", "bm25")
            distance = metadata.metadata.pop("distance", 1)

            logger.debug("db_name set to '%s'.", db_name)
            logger.debug("distance set to '%s'.", distance)

            # Sanitize metadata to ensure JSON serializability
            sanitized_metadata = self._sanitize_metadata(metadata.metadata)

            words = self._tokenizer.tokenize(text)
            document = self._data_models.Document.create(
                id=document_id,
                text=text,
                length=len(words),
                db_name=db_name,
                distance=distance,
                metadata=sanitized_metadata,  # Remaining metadata
            )
            for word in set(words):
                self._index_manager.update_term_index(
                    word, document.id, words.count(word)
                )
            self._index_manager.update_statistics(words, add=True)
        logger.info(
            "Document with metadata ID '%s' added successfully.", metadata.doc_id
        )

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        Sanitizes metadata to ensure it can be serialized to JSON.

        Args:
            metadata (dict): The metadata to sanitize.

        Returns:
            dict: The sanitized metadata.
        """
        sanitized = {}
        for key, value in metadata.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                # Convert unsupported types
                if isinstance(value, set):
                    sanitized[key] = list(value)  # Convert set to list
                elif hasattr(value, "__dict__"):
                    sanitized[key] = vars(value)  # Convert objects to dictionaries
                else:
                    sanitized[key] = str(value)  # Convert anything else to string
                logger.warning(
                    "Converted non-serializable metadata field '%s': %s -> %s",
                    key,
                    value,
                    sanitized[key],
                )
        return sanitized

    def remove_document(self, doc_id: str):
        """
        Removes a document from the repository by its ID.

        Args:
            doc_id (str): The document ID to remove.
        """
        logger.info("Removing document with ID '%s'.", doc_id)
        with self._db.atomic():
            document = self._data_models.Document.get_by_id(doc_id)
            words = self._tokenizer.tokenize(document.text)
            for word in set(words):
                self._index_manager.update_term_index(
                    word, doc_id, words.count(word), remove=True
                )

            document.delete_instance()
            self._index_manager.update_statistics(words, add=False)
        logger.info("Document with ID '%s' removed successfully.", doc_id)

    def update_document(self, text: str, metadata: BaseDocument):
        """
        Updates an existing document.

        Args:
            text (str): The updated content of the document.
            metadata (BaseDocument): The updated metadata for the document.
        """
        logger.info("Updating document with metadata ID '%s'.", metadata.doc_id)
        assert (
            "id" in metadata.__dict__
        ), "metadata should include 'id', but it's missing."
        id = metadata.id
        with self._db.atomic():
            self.remove_document(doc_id)
            self.add_document(text, metadata)
        logger.info(
            "Document with metadata ID '%s' updated successfully.", metadata.doc_id
        )

    def upsert_document(self, text: str, metadata: BaseDocument):
        """
        Adds or updates a document in the repository.

        Args:
            text (str): The content of the document.
            metadata (BaseDocument): The metadata associated with the document.
        """
        logger.info("Upserting document with metadata ID '%s'.", metadata.doc_id)
        try:
            self._data_models.Document.get_by_id(id)
            self.update_document(text, metadata)
        except self._data_models.Document.DoesNotExist:
            self.add_document(text, metadata)

    def remove_document_by_metadata(self, metadata: dict[str, Any]):
        """
        Removes documents matching specific metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to match for document removal.
        """
        logger.info("Removing documents by metadata: %s", metadata)

        with self._db.atomic():
            # Query to find documents that match all metadata key-value pairs
            query = self._data_models.Document.select()
            for key, value in metadata.items():
                query = query.where(
                    fn.json_extract(self._data_models.Document.metadata, "$." + key)
                    == value
                )

            # Iterate over matched documents and remove each one
            for document in query:
                words = self._tokenizer.tokenize(document.text)
                for word in set(words):
                    self._index_manager.update_term_index(
                        word, document.id, words.count(word), remove=True
                    )
                document.delete_instance()
                self._index_manager.update_statistics(words, add=False)
        logger.info("Documents matching metadata removed successfully.")
