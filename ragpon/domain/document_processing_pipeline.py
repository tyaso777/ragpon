import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd

from ragpon._utils.logging_helper import get_library_logger
from ragpon.chunk_processor import AbstractChunkProcessor
from ragpon.domain.document_reader import (
    AbstractDocumentReader,
    ExtensionBasedDocumentReaderFactory,
)
from ragpon.domain.domain import ChunkSourceInfo
from ragpon.domain.metadata_generator import CustomMetadataGenerator

# Initialize logger
logger = get_library_logger(__name__)


TMetadata = TypeVar("TMetadata", bound=Any)


class DocumentProcessingError(Exception):
    """Base class for exceptions in document processing pipelines."""

    pass


class FileProcessingError(DocumentProcessingError):
    """Raised when there is an error processing a file."""

    pass


class MetadataGenerationError(DocumentProcessingError):
    """Raised when there is an error generating metadata."""

    pass


class DataFrameProcessingError(DocumentProcessingError):
    """Raised when there is an error processing a DataFrame."""

    pass


class RowProcessingError(DataFrameProcessingError):
    """Raised when there is an error processing a specific row in a DataFrame."""

    pass


class AbstractDocumentProcessingPipeline(ABC, Generic[TMetadata]):
    """
    Abstract base class for document processing pipelines.

    Methods:
        process_document: Abstract method to process a document and extract chunks and metadata.
    """

    @abstractmethod
    def process_document(self, **kwargs) -> tuple[list[str], list[TMetadata]]:
        pass


class FilePathDocumentProcessingPipeline(
    AbstractDocumentProcessingPipeline[TMetadata], Generic[TMetadata]
):
    """
    Pipeline for processing files and generating document chunks and metadata.

    Attributes:
        chunk_processor (AbstractChunkProcessor): Processor for splitting text into chunks.
        metadata_generator (CustomMetadataGenerator[TMetadata]): Generator for metadata associated with chunks.
    """

    def __init__(
        self,
        chunk_processor: AbstractChunkProcessor,
        metadata_generator: CustomMetadataGenerator[TMetadata],
    ):
        self.chunk_processor = chunk_processor
        self.metadata_generator = metadata_generator

    def process_document(self, **kwargs: Any) -> tuple[list[str], list[TMetadata]]:
        """
        Processes a document by splitting it into chunks and generating metadata.

        Args:
            **kwargs: Arguments containing 'file_path'.

        Returns:
            tuple[list[str], list[TMetadata]]: A tuple containing chunks and their metadata.
        """
        start_page_number: int = kwargs.get("start_page_number", 1)
        file_path = kwargs.get("file_path")

        if not file_path or not Path(file_path).is_file():
            logger.error(f"Invalid or missing file path: {file_path}")
            raise FileProcessingError(f"Invalid or missing file path: {file_path}")

        logger.info(f"Starting document processing for file: {file_path}")

        try:
            reader: AbstractDocumentReader = (
                ExtensionBasedDocumentReaderFactory.get_document_reader(file_path)
            )
            pages: list[ChunkSourceInfo] = reader.read_document(
                start_page_number=start_page_number
            )
            logger.info(f"Extracted {len(pages)} pages from the document.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise FileProcessingError(f"File not found: {file_path}") from e
        except PermissionError as e:
            logger.error(f"Permission denied for file: {file_path}")
            raise FileProcessingError(f"Permission denied: {file_path}") from e
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            raise FileProcessingError(f"Error reading document: {file_path}") from e

        all_chunks: list[str] = []
        all_metadata: list[TMetadata] = []

        serial_number = 0
        for page in pages:
            try:
                logger.debug(
                    f"Processing page {page.metadata.get('page_number', 'unknown')}"
                )
                chunks = self.chunk_processor.process(page.text)
                for i, chunk in enumerate(chunks):
                    try:
                        logger.debug(
                            f"Processing chunk {i} on page {page.metadata.get('page_number', 'unknown')}"
                        )
                        meta = self.metadata_generator.generate(
                            doc_id=f"{Path(page.metadata.get('file_path', '')).stem}_No.{serial_number}",
                            text=chunk,  # TODO: ここでchunkを渡すのであれば、chunksとmetadataに分ける必要もないのでは…？
                            file_path=page.metadata.get("file_path", ""),
                            serial_number=serial_number,
                            page_number=page.metadata.get("page_number", 1),
                            chunk_index_in_page=i,
                        )
                        all_chunks.append(chunk)
                        all_metadata.append(meta)
                        serial_number += 1
                    except Exception as e:
                        logger.error(f"Error generating metadata for chunk {i}: {e}")
                        raise MetadataGenerationError(
                            f"Error generating metadata for chunk {i}"
                        ) from e
            except Exception as e:
                logger.error(
                    f"Error processing page {page.metadata.get('page_number', 'unknown')}: {e}"
                )

        logger.info(f"Document processing completed. Total chunks: {len(all_chunks)}")
        return all_chunks, all_metadata


class DataFrameDocumentProcessingPipeline(
    AbstractDocumentProcessingPipeline[TMetadata], Generic[TMetadata]
):
    """
    Pipeline for processing DataFrame objects and generating document chunks and metadata.

    Attributes:
        metadata_generator (CustomMetadataGenerator[TMetadata]): Generator for metadata associated with chunks.
        chunk_col_name (str): Name of the column containing text chunks.
        id_col_name (str): Name of the column containing unique identifiers.
    """

    def __init__(
        self,
        metadata_generator: CustomMetadataGenerator[TMetadata],
        chunk_col_name: str,
        id_col_name: str,
    ):
        self.metadata_generator = metadata_generator
        self.chunk_col_name = chunk_col_name
        self.id_col_name = id_col_name

    def process_document(self, **kwargs: Any) -> tuple[list[str], list[TMetadata]]:
        """
        Processes a DataFrame by extracting text chunks and generating metadata.

        Args:
            **kwargs: Arguments containing 'df' (the DataFrame to process).

        Returns:
            tuple[list[str], list[TMetadata]]: A tuple containing chunks and their metadata.
        """
        df: pd.DataFrame = kwargs.get("df")
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid or missing DataFrame.")
            raise DataFrameProcessingError("Invalid or missing DataFrame.")

        # Validate 'id_col_name' and 'chunk_col_name'
        if self.id_col_name not in df.columns:
            logger.error(
                f"Required column '{self.id_col_name}' is missing in the DataFrame."
            )
            raise DataFrameProcessingError(
                f"Required column '{self.id_col_name}' is missing in the DataFrame."
            )
        if self.chunk_col_name not in df.columns:
            logger.error(
                f"Required column '{self.chunk_col_name}' is missing in the DataFrame."
            )
            raise DataFrameProcessingError(
                f"Required column '{self.chunk_col_name}' is missing in the DataFrame."
            )

        # Check if 'id_col_name' values are unique when converted to strings
        str_ids = df[self.id_col_name].astype(str)
        if not str_ids.is_unique:
            logger.error(
                f"String-converted values in '{self.id_col_name}' must be unique."
            )
            raise DataFrameProcessingError(
                f"String-converted values in '{self.id_col_name}' must be unique."
            )

        logger.info("Starting DataFrame document processing.")

        all_chunks: list[str] = []
        all_metadata: list[TMetadata] = []

        for index, row in df.iterrows():
            try:
                logger.debug(f"Processing row {index}")
                chunk = row[self.chunk_col_name]
                metadata = self.metadata_generator.generate(
                    doc_id=str(row[self.id_col_name]),
                    **{**row.drop(self.id_col_name).to_dict(), "text": chunk},
                )
                all_chunks.append(chunk)
                all_metadata.append(metadata)
            except KeyError as e:
                logger.error(f"Missing column in row {index}: {e}")
                raise RowProcessingError(f"Missing column in row {index}") from e
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                raise RowProcessingError(f"Error processing row {index}") from e

        logger.info(f"DataFrame processing completed. Total rows: {len(df)}")
        return all_chunks, all_metadata
