from abc import ABC, abstractmethod

import spacy

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


class ChunkProcessingError(Exception):
    """Custom exception for errors during chunk processing."""

    pass


class AbstractChunkProcessor(ABC):
    @abstractmethod
    def process(self, text: str) -> list[str]:
        pass


class FixedLengthChunkProcessor(AbstractChunkProcessor):
    def __init__(self, chunk_size: int):
        """
        Initializes the FixedLengthChunkProcessor with a specified chunk size.

        Args:
            chunk_size (int): The size of each chunk in characters.
        """
        self.chunk_size = chunk_size

    def process(self, text: str) -> list[str]:
        """
        Splits text into fixed-length chunks.

        Args:
            text (str): The input text to split.

        Returns:
            list[str]: A list of text chunks.
        """
        if not text:
            logger.warning("Received empty text for processing.")
            return []

        try:
            chunks = []
            for start in range(0, len(text), self.chunk_size):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
            logger.info(f"Processed text into {len(chunks)} fixed-length chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error during fixed-length chunk processing: {e}")
            raise ChunkProcessingError("Failed to process fixed-length chunks.") from e


class NaturalChunkProcessor(AbstractChunkProcessor):
    def process(self, text: str) -> list[str]:
        """
        Splits text into chunks based on natural sentence delimiters.

        Args:
            text (str): The input text to split.

        Returns:
            list[str]: A list of sentence-based text chunks.
        """
        if not text:
            logger.warning("Received empty text for processing.")
            return []

        try:
            chunks = []
            current_chunk = ""
            for char in text:
                current_chunk += char
                if char in (".", "。", "！", "？", "!", "?"):
                    chunks.append(current_chunk)
                    current_chunk = ""
            if current_chunk:
                chunks.append(current_chunk)
            logger.info(f"Processed text into {len(chunks)} natural chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error during natural chunk processing: {e}")
            raise ChunkProcessingError("Failed to process natural chunks.") from e


class SingleChunkProcessor(AbstractChunkProcessor):
    def process(self, text: str) -> list[str]:
        """
        Returns the entire text as a single chunk.

        Args:
            text (str): The input text.

        Returns:
            list[str]: A single-element list containing the input text.
        """
        if not text:
            logger.warning("Received empty text for processing.")
            return []

        try:
            logger.info("Processed text into a single chunk.")
            return [text]
        except Exception as e:
            logger.error(f"Error during single chunk processing: {e}")
            raise ChunkProcessingError("Failed to process single chunk.") from e


class JAGinzaChunkProcessor(AbstractChunkProcessor):
    def __init__(self, chunk_size: int = 100):
        """
        Initializes the JAGinzaChunkProcessor with a specified chunk size.

        Args:
            chunk_size (int, optional): The size of each chunk in characters. Defaults to 100.
        """
        try:
            self.nlp = spacy.load("ja_ginza")
            self.chunk_size = chunk_size
            logger.info("Initialized JAGinzaChunkProcessor.")
        except Exception as e:
            logger.error(f"Error initializing JAGinzaChunkProcessor: {e}")
            raise ChunkProcessingError(
                "Failed to initialize JAGinzaChunkProcessor."
            ) from e

    def process(self, text: str) -> list[str]:
        """
        Splits text into chunks using the JAGinza NLP model.

        Args:
            text (str): The input text to split.

        Returns:
            list[str]: A list of text chunks.
        """
        if not text:
            logger.warning("Received empty text for processing.")
            return []

        try:
            chunks: list[str] = []
            current_chunk: list[str] = []
            current_size: int = 0
            for span in self.nlp(text).sents:
                strspan = str(span)
                if current_chunk and current_size + len(strspan) > self.chunk_size:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(strspan)
                current_size += len(strspan)
            if current_chunk:
                chunks.append("".join(current_chunk))
            logger.info(f"Processed text into {len(chunks)} chunks using JAGinza.")
            return chunks
        except Exception as e:
            logger.error(f"Error during JAGinza chunk processing: {e}")
            raise ChunkProcessingError("Failed to process chunks using JAGinza.") from e
