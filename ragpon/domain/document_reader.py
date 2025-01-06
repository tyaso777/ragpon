import os
from abc import ABC, abstractmethod

import chardet
import docx
import fitz
import pypdf
from bs4 import BeautifulSoup

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import ChunkSourceInfo

# Initialize logger
logger = get_library_logger(__name__)


class AbstractDocumentReader(ABC):
    """
    Abstract base class for document readers.

    Attributes:
        filepath (str): Path to the file to be read.
    """

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath

    @abstractmethod
    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        """
        Reads the document and extracts its contents as chunks.

        Args:
            start_page_number (int): The starting page number.

        Returns:
            list[ChunkSourceInfo]: List of extracted text chunks with metadata.
        """
        pass


class PDFReaderPyPDF(AbstractDocumentReader):
    """
    Document reader for PDF files using PyPDF.
    """

    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        pages_text = []
        try:
            reader = pypdf.PdfReader(self.filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                pages_text.append(
                    ChunkSourceInfo(
                        text=text,
                        unit_index=page_num + start_page_number,
                        metadata={
                            "file_path": self.filepath,
                            "page_number": page_num + start_page_number,
                        },
                    )
                )
            logger.debug(f"Read {len(pages_text)} pages from PDF file.")
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise IOError(f"Error reading PDF file: {e}")
        return pages_text


class PDFReaderPyMuPDF(AbstractDocumentReader):
    """
    Document reader for PDF files using PyMuPDF.
    """

    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        pages_text = []
        try:
            document = fitz.open(self.filepath)
            for page_num in range(len(document)):
                page = document[page_num]
                text = page.get_text("text")
                pages_text.append(
                    ChunkSourceInfo(
                        text=text,
                        unit_index=page_num + start_page_number,
                        metadata={
                            "file_path": self.filepath,
                            "page_number": page_num + start_page_number,
                        },
                    )
                )
            logger.debug(f"Read {len(pages_text)} pages from PDF file using PyMuPDF.")
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise IOError(f"Error reading PDF file: {e}")
        return pages_text


class TXTReader(AbstractDocumentReader):
    """
    Document reader for plain text files.
    """

    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        try:
            with open(self.filepath, "rb") as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)["encoding"]
            with open(self.filepath, "r", encoding=encoding) as file:
                text = file.read()
            logger.debug(f"Read text file with encoding {encoding}.")
            return [
                ChunkSourceInfo(
                    text=text,
                    unit_index=start_page_number,
                    metadata={
                        "file_path": self.filepath,
                        "page_number": start_page_number,
                    },
                )
            ]
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise FileNotFoundError(f"File not found: {self.filepath}")
        except IOError as e:
            logger.error(f"Error reading file: {e}")
            raise IOError(f"Error reading file: {self.filepath}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise Exception(f"An error occurred: {e}")


class HTMLReader(AbstractDocumentReader):
    """
    Document reader for HTML files.
    """

    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        try:
            with open(self.filepath, "rb") as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)["encoding"]
            with open(self.filepath, "r", encoding=encoding) as file:
                soup = BeautifulSoup(file, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
            logger.debug(f"Read HTML file with encoding {encoding}.")
            return [
                ChunkSourceInfo(
                    text=text,
                    unit_index=start_page_number,
                    metadata={
                        "file_path": self.filepath,
                        "page_number": start_page_number,
                    },
                )
            ]
        except FileNotFoundError:
            logger.error(f"File not found: {self.filepath}")
            raise FileNotFoundError(f"File not found: {self.filepath}")
        except IOError as e:
            logger.error(f"Error reading file: {e}")
            raise IOError(f"Error reading file: {self.filepath}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise Exception(f"An error occurred: {e}")


class WordReader(AbstractDocumentReader):
    """
    Document reader for Word files (.docx).
    """

    def read_document(self, start_page_number: int = 1) -> list[ChunkSourceInfo]:
        paragraphs_text = []
        try:
            document = docx.Document(self.filepath)
            for para_num, paragraph in enumerate(document.paragraphs):
                text = paragraph.text.strip()
                if text:  # Skip empty paragraphs
                    paragraphs_text.append(
                        ChunkSourceInfo(
                            text=text,
                            unit_index=para_num + start_page_number,
                            metadata={
                                "file_path": self.filepath,
                                "paragraph_number": para_num + start_page_number,
                            },
                        )
                    )
            logger.debug(f"Read {len(paragraphs_text)} paragraphs from Word file.")
        except Exception as e:
            logger.error(f"Error reading Word file: {e}")
            raise IOError(f"Error reading Word file: {e}")
        return paragraphs_text


class AbstractDocumentReaderFactory(ABC):
    """
    Abstract factory for creating document readers.
    """

    @classmethod
    @abstractmethod
    def get_document_reader(cls, file_path: str) -> AbstractDocumentReader:
        """
        Returns an appropriate document reader for the given file path.

        Args:
            file_path (str): Path to the file.

        Returns:
            AbstractDocumentReader: An instance of a document reader.
        """
        pass


class ExtensionBasedDocumentReaderFactory(AbstractDocumentReaderFactory):
    """
    Factory for creating document readers based on file extensions.
    """

    @classmethod
    def get_document_reader(cls, file_path: str) -> AbstractDocumentReader:
        if file_path.lower().endswith(".pdf"):
            logger.debug("Creating PDFReaderPyMuPDF for file.")
            return PDFReaderPyMuPDF(file_path)
        elif file_path.lower().endswith(".txt"):
            logger.debug("Creating TXTReader for file.")
            return TXTReader(file_path)
        elif file_path.lower().endswith(".html"):
            logger.debug("Creating HTMLReader for file.")
            return HTMLReader(file_path)
        elif file_path.lower().endswith(".docx"):
            logger.debug("Creating WordReader for file.")
            return WordReader(file_path)
        else:
            logger.error("Unsupported file type.")
            raise ValueError("Unsupported file type")
