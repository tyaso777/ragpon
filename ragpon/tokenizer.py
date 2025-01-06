import re
from abc import ABC, abstractmethod

import sudachipy.dictionary
import sudachipy.tokenizer

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


class AbstractTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Methods:
        tokenize(document: str) -> list[str]: Tokenizes the input document into a list of tokens.
    """

    @abstractmethod
    def tokenize(self, document: str) -> list[str]:
        pass


class SudachiTokenizer(AbstractTokenizer):
    """
    Tokenizer using SudachiPy for Japanese text analysis.

    Attributes:
        _tokenizer_obj: The Sudachi tokenizer instance.
        _low_value_words (set[str]): A set of low-value words to exclude from the results.
    """

    def __init__(self):
        """
        Initializes the SudachiTokenizer with a tokenizer instance and a list of low-value words.
        """
        logger.info("Initializing SudachiTokenizer.")
        # Initialize Sudachi tokenizer
        self._tokenizer_obj = sudachipy.dictionary.Dictionary().create()
        # List of low-information value words (as an example)
        self._low_value_words = {
            "する",
            "ある",
            "いる",
            "居る",
            "為る",
            "なる",
            "やる",
            "持つ",
            "行う",
        }
        logger.info("SudachiTokenizer initialized successfully.")

    def _is_important_pos(self, part_of_speech):
        """
        Determines if the part of speech is important.

        Args:
            part_of_speech (list[str]): The part of speech tags for a token.

        Returns:
            bool: True if the part of speech is important, False otherwise.
        """
        # Part of speech filter
        is_important = part_of_speech[0] in {"名詞", "動詞", "形容詞"}
        logger.debug(
            "Part of speech '%s' is important: %s", part_of_speech, is_important
        )
        return is_important

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text into a list of important words.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str]: A list of tokens after filtering and normalization.
        """
        logger.info("Tokenizing text of length: %d", len(text))
        # Convert alphanumeric characters to lowercase
        text = re.sub(r"[A-Za-z0-9]+", lambda m: m.group(0).lower(), text)
        logger.debug(
            "Text after lowercasing alphanumerics: %s", text[:50]
        )  # Show only first 50 chars
        # Morphological analysis
        tokens = self._tokenizer_obj.tokenize(
            text, sudachipy.tokenizer.Tokenizer.SplitMode.C
        )
        result = []
        for token in tokens:
            pos = token.part_of_speech()
            if self._is_important_pos(pos):
                # Normalize synonyms to their base form
                base_form = token.normalized_form()
                # Exclude low-information value words
                if base_form not in self._low_value_words:
                    result.append(base_form)
        logger.info("Tokenization completed. Number of tokens: %d", len(result))
        return result
