from datetime import date
from typing import Any, Type, TypeVar

from peewee import (
    BigIntegerField,
    CharField,
    CompositeKey,
    DateField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)
from playhouse.sqlite_ext import JSONField

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document

# Initialize logger
logger = get_library_logger(__name__)

# Type Aliases
TMetadata = TypeVar("TMetadata", bound=BaseDocument)
TResult = TypeVar("TResult", bound=Document)


# Base Model Definition
class BaseDocumentModel(Model):
    """Defines the base document model for the database.

    Attributes:
        id (CharField): Primary key of the document.
        text (TextField): The content of the document.
        length (IntegerField): The length of the text.
    """

    id = CharField(primary_key=True)
    text = TextField()
    length = IntegerField()

    class Meta:
        pass


def create_document_model(
    db: SqliteDatabase, schema: Type[TMetadata], base_document_model: Type[Model]
) -> Type[Model]:
    """Creates a document model dynamically based on a schema.

    Args:
        db (SqliteDatabase): The database connection.
        schema (Type[TMetadata]): The schema class that defines the document structure.
        base_document_model (Type[Model]): The base Peewee model to extend.

    Returns:
        Type[Model]: A dynamically created document model.

    Raises:
        ValueError: If an unsupported field type is encountered.
    """
    logger.info("Creating document model.")
    attributes: dict[str, Any] = {"__module__": __name__}
    for field_name, field_type in schema.__annotations__.items():
        if field_name == "doc_id":
            attributes["id"] = CharField(primary_key=True)
            logger.debug("Mapped 'doc_id' to 'id' with primary key.")
        elif field_type is str:
            attributes[field_name] = TextField()
        elif field_type is int:
            attributes[field_name] = BigIntegerField()
        elif field_type is float:
            attributes[field_name] = FloatField()
        elif field_type is date:
            attributes[field_name] = DateField()
        elif field_name == "metadata":
            attributes["metadata"] = JSONField()
        else:
            logger.error("Unsupported type for field '%s': %s", field_name, field_type)
            raise ValueError(f"Unsupported type: {field_type}")
    model_class = type("Document", (base_document_model,), attributes)
    model_class._meta.database = db
    logger.info("Document model created successfully.")
    return model_class


def create_statistics_model(db: SqliteDatabase) -> Type[Model]:
    """Creates a model for tracking document statistics.

    Args:
        db (SqliteDatabase): The database connection.

    Returns:
        Type[Model]: A model for tracking document statistics.
    """
    logger.info("Creating statistics model.")

    class Statistics(Model):
        total_documents = BigIntegerField()
        total_words = BigIntegerField()

        class Meta:
            database = db

    logger.info("Statistics model created successfully.")
    return Statistics


def create_term_model(db: SqliteDatabase) -> Type[Model]:
    """Creates a model for managing terms and their frequencies.

    Args:
        db (SqliteDatabase): The database connection.

    Returns:
        Type[Model]: A model for managing terms.
    """
    logger.info("Creating term model.")

    class Term(Model):
        term = CharField(primary_key=True)
        document_frequency = IntegerField()

        class Meta:
            database = db

    logger.info("Term model created successfully.")
    return Term


def create_term_frequency_model(
    db: SqliteDatabase, term_model: Type[Model], document_model: Type[Model]
) -> Type[Model]:
    """Creates a model for tracking term frequencies in documents.

    Args:
        db (SqliteDatabase): The database connection.
        term_model (Type[Model]): The model representing terms.
        document_model (Type[Model]): The model representing documents.

    Returns:
        Type[Model]: A model for tracking term frequencies.
    """
    logger.info("Creating term frequency model.")

    class TermFrequency(Model):
        term = ForeignKeyField(term_model, backref="frequency")
        document_id = ForeignKeyField(document_model, backref="frequency")
        term_frequency = IntegerField()

        class Meta:
            primary_key = CompositeKey("term", "document_id")
            indexes = (
                (("term",), False),
                (("document_id",), False),
            )
            database = db

    logger.info("Term frequency model created successfully.")
    return TermFrequency


class DataModels:
    """Manages data models for the BM25 repository.

    Attributes:
        Document (Type[Model]): The dynamically created document model.
        Statistics (Type[Model]): The model for tracking statistics.
        Term (Type[Model]): The model for managing terms.
        TermFrequency (Type[Model]): The model for tracking term frequencies.
    """

    def __init__(self, db: SqliteDatabase, schema: Type[TMetadata]) -> None:
        """Initializes the DataModels instance.

        Args:
            db (SqliteDatabase): The database connection.
            schema (Type[TMetadata]): The schema defining the document structure.
        """
        logger.info("Initializing DataModels.")
        self._db = db
        self.Document = create_document_model(
            db=self._db, schema=schema, base_document_model=BaseDocumentModel
        )
        self.Statistics = create_statistics_model(db=self._db)
        self.Term = create_term_model(db=self._db)
        self.TermFrequency = create_term_frequency_model(
            db=self._db, term_model=self.Term, document_model=self.Document
        )
        self._create_tables()
        logger.info("DataModels initialized successfully.")

    def _create_tables(self) -> None:
        """Creates the required tables in the database."""
        try:
            logger.info("Creating tables in the database.")
            self._db.create_tables(
                [
                    self.Statistics,
                    self.Document,
                    self.Term,
                    self.TermFrequency,
                ],
                safe=True,
            )
            logger.info("Tables created successfully.")
        except Exception as e:
            logger.error("Error creating tables: %s", str(e))
            raise
