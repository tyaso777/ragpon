# %%
from dataclasses import dataclass, field
from typing import Any

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


@dataclass
class ChunkSourceInfo:
    """
    Represents the source information for a chunk of text.

    Attributes:
        text (str): The text content of the chunk.
        unit_index (int): The index of the unit within the source.
        metadata (dict[str, Any]): Additional metadata associated with the chunk.
    """

    text: str
    unit_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Logs the initialization of a ChunkSourceInfo instance."""
        logger.debug(
            "Initialized ChunkSourceInfo with unit_index: %d, text length: %d",
            self.unit_index,
            len(self.text),
        )


@dataclass
class BaseDocument:
    """
    Represents the basic structure of a document.

    Attributes:
        doc_id (str): Unique identifier for the document, typically formatted as "{filename}_No.{serial_num}".
        text (str): The main content of the document.
        db_name (str): Name of the database where the document is stored. Defaults to "default_db".
        distance (float): A similarity or distance score associated with the document. Defaults to 1.0.
        metadata (dict[str, Any]): Additional metadata associated with the document. Defaults to an empty dictionary.
    """

    doc_id: str  # Format: f"{filename}_No.{serial_num}"
    text: str
    db_name: str = "default"
    distance: float = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Logs the initialization of a BaseDocument instance."""
        logger.debug(
            "Initialized BaseDocument with doc_id: %s, text length: %d",
            self.doc_id,
            len(self.text),
        )


@dataclass
class Document:
    """
    Represents a document that includes optional enhanced and reranked information.

    Attributes:
        base_document (BaseDocument): The core details of the document.
        enhanced_text ([str | None]): Additional enhanced information, if available.
        rerank (float | None): Reranking information, if available.
    """

    base_document: BaseDocument
    enhanced_text: str | None = None
    rerank: float | None = None

    def __post_init__(self):
        """Logs the initialization of a Document instance."""
        logger.debug(
            "Initialized Document with doc_id: %s, enhanced_text: %s, rerank: %s",
            self.base_document.doc_id,
            "present" if self.enhanced_text else "none",
            "present" if self.rerank else "none",
        )


# %%
