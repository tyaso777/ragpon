# %%
import os
from abc import ABC, abstractmethod
from typing import Sequence, Union

import ctranslate2
import numpy as np
import torch.nn.functional as F
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ragpon._utils.logging_helper import get_library_logger
from ragpon.config import Config

# Initialize logger
logger = get_library_logger(__name__)

# ref: Japanese embedding model leaderboard: https://github.com/sbintuitions/JMTEB/blob/main/leaderboard.md


class AbstractEmbeddingModel(ABC):
    @abstractmethod
    def __init__(
        self,
        config: Config,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        """
        Initialize the large language model with a configuration.

        Args:
            config (Config): Configuration object to load model settings.
        """
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        pass

    @abstractmethod
    def embed_single(self, text: str, use_prefix: str = "None") -> list[list[float]]:
        pass

    @abstractmethod
    def embed_batch(
        self, texts: list[str], use_prefix: str = "None"
    ) -> list[list[float]]:
        pass


class MultilingualE5LargeEmbedder(AbstractEmbeddingModel):
    def __init__(
        self,
        config: Config,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        """
        Initializes the MultilingualE5LargeEmbedder using a provided configuration.

        Args:
            config (Config): Configuration object to load model paths.
            query_prefix (str): Prefix to prepend to queries.
            passage_prefix (str): Prefix to prepend to passages.
        """
        try:
            model_path = config.get("MODELS.MULTILINGUAL_E5_LARGE_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "Model path for Multilingual E5 Large is not set in the configuration."
                )

            logger.info(
                "Initializing MultilingualE5LargeEmbedder with model path: %s",
                model_path,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_safetensors=True
            )
            self.model = AutoModel.from_pretrained(model_path, use_safetensors=True)
            self.query_prefix = query_prefix
            self.passage_prefix = passage_prefix

            logger.info("MultilingualE5LargeEmbedder initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MultilingualE5LargeEmbedder: {e}")
            raise

    def _average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        Applies average pooling on hidden states based on the attention mask.

        Args:
            last_hidden_states (Tensor): Hidden states from the transformer model.
            attention_mask (Tensor): Attention mask indicating non-padding tokens.

        Returns:
            Tensor: Averaged pooled representation.
        """
        try:
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            pooled = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            logger.debug("Average pooling completed successfully.")
            return pooled
        except Exception as e:
            logger.error("Error during average pooling: %s", str(e))
            raise

    def embed_single(self, text: str, use_prefix: str = "None") -> list[list[float]]:
        """
        Embeds a single piece of text into a dense vector representation.

        Args:
            text (str): The input text to embed.
            use_prefix (str): Specifies which prefix to use. Options are:
                            - "query": Use the query prefix.
                            - "passage": Use the passage prefix.
                            - "None": Do not use any prefix (default).

        Returns:
            list[list[float]]: Dense vector representation of the input text.
        """
        try:
            # Validate use_prefix
            if use_prefix not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix. Choose from 'query', 'passage', or 'None'."
                )

            # Add the selected prefix if applicable
            if use_prefix == "query":
                text = self.query_prefix + text
            elif use_prefix == "passage":
                text = self.passage_prefix + text

            logger.info(f"Embedding single text input: {text}")
            batch_dict = self.tokenizer(
                text=[text],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1).tolist()
            logger.info("Single text embedded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding single text: {e}")
            raise

    def embed_batch(
        self, texts: list[str], use_prefix: str = "None"
    ) -> list[list[float]]:
        """
        Embeds a batch of text inputs into dense vector representations.

        Args:
            texts (list[str]): A list of input texts to embed.
            use_prefix (str): Specifies which prefix to use. Options are:
                            - "query": Use the query prefix.
                            - "passage": Use the passage prefix.
                            - "None": Do not use any prefix (default).

        Returns:
            list[list[float]]: Dense vector representations of the input texts.
        """
        try:
            # Validate inputs
            if not isinstance(texts, list) or not all(
                isinstance(item, str) for item in texts
            ):
                raise ValueError("Input must be a list of strings")

            if use_prefix not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix. Choose from 'query', 'passage', or 'None'."
                )

            logger.info("Embedding batch of texts with prefix: %s", use_prefix)

            # Apply the selected prefix to each text if applicable
            if use_prefix == "query":
                texts = [self.query_prefix + text for text in texts]
            elif use_prefix == "passage":
                texts = [self.passage_prefix + text for text in texts]

            logger.debug(f"Texts after prefix application: {texts}")

            # Tokenize and process embeddings
            batch_dict = self.tokenizer(
                text=texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1).tolist()
            logger.info("Batch text embedding completed successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch of texts: {e}")
            raise


class RuriLargeEmbedder(AbstractEmbeddingModel):
    def __init__(
        self,
        config: Config,
        query_prefix: str = "クエリ: ",
        passage_prefix: str = "文章: ",
    ):
        """
        Initializes the RuriLargeEmbedder using the "cl-nagoya/ruri-large" model.

        Args:
            config (Config): Configuration object to load model paths.
            query_prefix (str): Prefix to prepend to queries.
            passage_prefix (str): Prefix to prepend to passages.
        """
        try:
            model_path = config.get("MODELS.CL_NAGOYA_RURI_LARGE_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "Model path for cl-nagoya-ruri-large is not set in the configuration."
                )

            logger.info(
                "Initializing RuriLargeEmbedder with model path: %s",
                model_path,
            )

            self.model = SentenceTransformer(model_path)
            self.query_prefix = query_prefix
            self.passage_prefix = passage_prefix
            logger.info("RuriLargeEmbedder initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RuriLargeEmbedder: {e}")
            raise

    def embed_single(self, text: str, use_prefix: str = "None") -> list[list[float]]:
        """
        Embeds a single text input into a dense vector representation.

        Args:
            text (str): Input text to embed.
            use_prefix (str): Specifies which prefix to use. Options are:
                              - "query": Use the query prefix.
                              - "passage": Use the passage prefix.
                              - "None": Do not use any prefix (default).

        Returns:
            list[list[float]]: Dense vector representation of the input text.
        """
        try:
            # Validate use_prefix
            if use_prefix not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix. Choose from 'query', 'passage', or 'None'."
                )

            # Add the selected prefix if applicable
            if use_prefix == "query":
                text = self.query_prefix + text
            elif use_prefix == "passage":
                text = self.passage_prefix + text

            logger.info(f"Embedding single text input: {text}")
            embedding = self.model.encode([text]).tolist()
            logger.info("Single text embedded successfully.")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding single text: {e}")
            raise

    def embed_batch(
        self, texts: list[str], use_prefix: str = "None"
    ) -> list[list[float]]:
        """
        Embeds a batch of text inputs into dense vector representations.

        Args:
            texts (list[str]): List of input texts to embed.
            use_prefix (str): Specifies which prefix to use. Options are:
                              - "query": Use the query prefix.
                              - "passage": Use the passage prefix.
                              - "None": Do not use any prefix (default).

        Returns:
            list[list[float]]: Dense vector representations of the input texts.
        """
        try:
            # Validate inputs
            if not isinstance(texts, list) or not all(
                isinstance(item, str) for item in texts
            ):
                raise ValueError("Input must be a list of strings")

            if use_prefix not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix. Choose from 'query', 'passage', or 'None'."
                )

            logger.info("Embedding batch of texts with prefix: %s", use_prefix)

            # Apply the selected prefix to each text if applicable
            if use_prefix == "query":
                texts = [self.query_prefix + text for text in texts]
            elif use_prefix == "passage":
                texts = [self.passage_prefix + text for text in texts]

            logger.debug(f"Texts after prefix application: {texts}")

            embeddings = self.model.encode(texts).tolist()
            logger.info("Batch text embedding completed successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch of texts: {e}")
            raise


class RuriLargeEmbedderCTranslate2(AbstractEmbeddingModel):
    def __init__(
        self,
        config: Config,
        query_prefix: str = "クエリ: ",
        passage_prefix: str = "文章: ",
    ):
        """
        Initializes the RuriLargeEmbedder using the CTranslate2 converted version of "cl-nagoya/ruri-large".

        Args:
            config (Config): Configuration object to load model paths.
            query_prefix (str): Prefix to prepend to queries.
            passage_prefix (str): Prefix to prepend to passages.
        """
        try:
            base_model_path = config.get("MODELS.CL_NAGOYA_RURI_LARGE_MODEL_PATH")
            model_path = os.path.join(base_model_path, "ct2-model")

            if not base_model_path:
                raise ValueError("Model path for cl-nagoya-ruri-large is not set.")

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.model = ctranslate2.Encoder(model_path, device="cpu")

            self.query_prefix = query_prefix
            self.passage_prefix = passage_prefix

            logger.info("RuriLargeEmbedderCTranslate2 initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RuriLargeEmbedderCTranslate2: {e}")
            raise

    def embed_single(self, text: str, use_prefix: str = "None") -> list[list[float]]:
        """
        Compute embedding for a single text input.

        Args:
            text (str): Text to embed.
            use_prefix (str): Specifies which prefix to use. Options:
                              - "query": Use query prefix.
                              - "passage": Use passage prefix.
                              - "None": No prefix.

        Returns:
            list[list[float]]: Embedded representation.
        """
        try:
            if use_prefix == "query":
                text = self.query_prefix + text
            elif use_prefix == "passage":
                text = self.passage_prefix + text

            inputs = self.tokenizer(
                [text], padding=True, truncation=True, return_tensors="np"
            )
            input_ids = inputs["input_ids"].tolist()
            outputs = self.model.forward_batch(input_ids)
            embedding = np.mean(outputs.last_hidden_state, axis=1).tolist()

            logger.info("Single text embedded successfully using CTranslate2.")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding single text: {e}")
            raise

    def embed_batch(
        self, texts: list[str], use_prefix: str = "None"
    ) -> list[list[float]]:
        """
        Embed a batch of texts using the CTranslate2 embedding model.

        Args:
            texts (list[str]): Texts to embed.
            use_prefix (str): Specifies which prefix to use. Options:
                              - "query": Use query prefix.
                              - "passage": Use passage prefix.
                              - "None": No prefix.

        Returns:
            list[list[float]]: List of embeddings.
        """
        try:
            if use_prefix == "query":
                texts = [self.query_prefix + t for t in texts]
            elif use_prefix == "passage":
                texts = [self.passage_prefix + t for t in texts]

            inputs = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="np"
            )
            input_ids = inputs["input_ids"].tolist()
            outputs = self.model.forward_batch(input_ids)
            embeddings = np.mean(outputs.last_hidden_state, axis=1).tolist()

            logger.info("Batch embedding with CTranslate2 completed successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch embedding with CTranslate2: {e}")
            raise


class ChromaDBEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, embedding_model: AbstractEmbeddingModel):
        """
        Initializes the ChromaDBEmbeddingAdapter with a given embedding model.

        Args:
            embedding_model (AbstractEmbeddingModel): The embedding model to use.
        """
        self.embedding_model = embedding_model
        self.query_prefix = embedding_model.query_prefix
        self.passage_prefix = embedding_model.passage_prefix

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds input documents into dense vector representations.

        Args:
            input (Documents): Input documents to embed.

        Returns:
            list[list[float]]: Dense vector representations of the documents.
        """
        try:
            logger.info("Embedding documents with ChromaDBEmbeddingAdapter.")
            embeddings: list[Union[Sequence[float], Sequence[int]]] = []
            if isinstance(input, str):
                embeddings = self.embedding_model.embed_single(input)
            elif isinstance(input, list) and all(
                isinstance(item, str) for item in input
            ):
                embeddings = self.embedding_model.embed_batch(input)
            else:
                raise ValueError("Input must be a string or a list of strings")
            logger.info("Documents embedded successfully.")
            return embeddings
        except Exception as e:
            logger.error("Error embedding documents: %s", str(e))
            raise


class AbstractSimilarityCalculator(ABC):
    @abstractmethod
    def calculate_similarity(self, text0: str, text1: str) -> float:
        """
        Calculates the similarity between two pieces of text.

        Args:
            text0 (str): The first piece of text.
            text1 (str): The second piece of text.

        Returns:
            float: The calculated similarity score.
        """
        pass


class CosineSimilarityCalculator(AbstractSimilarityCalculator):
    def __init__(self, embedding_model: AbstractEmbeddingModel):
        """
        Initializes the CosineSimilarityCalculator with a given embedding model.

        Args:
            embedding_model (AbstractEmbeddingModel): The embedding model to use for vectorization.
        """
        self.embedding_model = embedding_model

    def _cosine_similarity(self, vec0: list[float], vec1: list[float]) -> float:
        """
        Computes the cosine similarity between two vectors.

        Args:
            vec0 (list[float]): The first vector.
            vec1 (list[float]): The second vector.

        Returns:
            float: The cosine similarity score.
        """
        try:
            dot_product = np.dot(vec0, vec1)
            norm_vec0 = np.linalg.norm(vec0)
            norm_vec1 = np.linalg.norm(vec1)
            similarity = dot_product / (norm_vec0 * norm_vec1)
            logger.debug("Cosine similarity computed successfully.")
            return similarity
        except Exception as e:
            logger.error("Error computing cosine similarity: %s", str(e))
            raise

    def calculate_similarity(
        self,
        text0: str,
        text1: str,
        use_prefix0: str = "None",
        use_prefix1: str = "None",
    ) -> float:
        """
        Calculates the cosine similarity between two pieces of text.

        Args:
            text0 (str): The first piece of text.
            text1 (str): The second piece of text.
            use_prefix0 (str): Prefix for the first text. Options are 'query', 'passage', or 'None'.
            use_prefix1 (str): Prefix for the second text. Options are 'query', 'passage', or 'None'.

        Returns:
            float: The calculated cosine similarity score.
        """
        try:
            # Validate prefixes
            if use_prefix0 not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix0. Choose from 'query', 'passage', or 'None'."
                )
            if use_prefix1 not in {"query", "passage", "None"}:
                raise ValueError(
                    "Invalid use_prefix1. Choose from 'query', 'passage', or 'None'."
                )

            logger.info(
                f"Calculating similarity between text0 (prefix: {use_prefix0}) and text1 (prefix: {use_prefix1})"
            )

            # Apply prefixes and embed texts
            vec0 = self.embedding_model.embed_single(text0, use_prefix=use_prefix0)[0]
            vec1 = self.embedding_model.embed_single(text1, use_prefix=use_prefix1)[0]

            # Compute similarity
            similarity = self._cosine_similarity(vec0, vec1)
            logger.info(f"Similarity calculated successfully: {similarity:.6f}")
            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise


# %%
