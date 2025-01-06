# %%
from abc import ABC, abstractmethod

import torch
from sentence_transformers import CrossEncoder

from ragpon._utils.logging_helper import get_library_logger
from ragpon.config import Config
from ragpon.domain.domain import Document

# Initialize logger
logger = get_library_logger(__name__)


class AbstractRelevanceEvaluator(ABC):
    """
    Abstract base class for relevance evaluators.
    Provides methods for single and batch scoring of question-answer pairs.
    """

    @abstractmethod
    def score_single(self, question: str, answer: str) -> list[float]:
        """
        Compute a relevance score for a single question-answer pair.

        Args:
            question (str): The input question.
            answer (str): The corresponding answer.

        Returns:
            list[float]: The relevance score.
        """
        pass

    def score_batch(self, question: str, answers: list[str]) -> list[float]:
        """
        Compute relevance scores for a batch of question-answer pairs.

        Args:
            question (str): The input question.
            answers (list[str]): A list of corresponding answers.

        Returns:
            list[float]: A list of relevance scores.
        """
        pass


class JapaneseRerankerCrossEncoderLargeV1Evaluator(AbstractRelevanceEvaluator):
    """
    Relevance evaluator using a CrossEncoder model for Japanese text.
    """

    def __init__(self, config: Config):
        """
        Initialize the evaluator with a CrossEncoder model.

        Args:
            config (Config): Configuration object for retrieving model paths.

        Raises:
            ValueError: If the model path is not set in the configuration.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = config.get(
                "MODELS.JAPANESE_RERANKER_CROSS_ENCODER_LARGE_V1_PATH"
            )
            if not model_path:
                raise ValueError(
                    "Model path for Japanese Reranker is not set in the configuration."
                )

            # Initialize the model
            self.model = CrossEncoder(model_path, max_length=512, device=device)
            if device == "cuda":
                self.model.model.half()
            logger.info(
                "JapaneseRerankerCrossEncoderLargeV1Evaluator initialized successfully."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize JapaneseRerankerCrossEncoderLargeV1Evaluator: {e}"
            )
            raise

    def score_single(self, question: str, answer: str) -> list[float]:
        """
        Compute a relevance score for a single question-answer pair.

        Args:
            question (str): The input question.
            answer (str): The corresponding answer.

        Returns:
            list{float}: The relevance score.
        """
        try:
            score = self.model.predict([(question, answer)]).tolist()
            logger.debug(f"Single score computed: {score}")
            return score
        except Exception as e:
            logger.error(f"Error during single score computation: {e}")
            raise

    def score_batch(self, question: str, answers: list[str]) -> list[float]:
        """
        Compute relevance scores for a batch of question-answer pairs.

        Args:
            question (str): The input question.
            answers (list[str]): A list of corresponding answers.

        Returns:
            list[float]: A list of relevance scores.
        """
        try:
            qa_comb = [(question, answer) for answer in answers]
            scores = self.model.predict(qa_comb).tolist()
            logger.debug(f"Batch scores computed: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error during batch score computation: {e}")
            raise


class RuriRerankerLargeEvaluator(AbstractRelevanceEvaluator):
    """
    Relevance evaluator using the Ruri Reranker Large model.
    """

    def __init__(self, config: Config):
        """
        Initialize the evaluator with the Ruri Reranker Large model.

        Args:
            config (Config): Configuration object for retrieving model paths.

        Raises:
            ValueError: If the model path is not set in the configuration.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = config.get("MODELS.RURI_RERANKER_LARGE_PATH")
            if not model_path:
                raise ValueError(
                    "Model path for Ruri Reranker is not set in the configuration."
                )

            self.model = CrossEncoder(model_path, device=device)
            if device == "cuda":
                self.model.model.half()
            logger.info(
                f"RuriRerankerLargeEvaluator {model_path} initialized successfully."
            )
        except Exception as e:
            logger.error(f"Failed to initialize RuriRerankerLargeEvaluator: {e}")
            raise

    def score_single(self, question: str, answer: str) -> list[float]:
        """
        Compute a relevance score for a single question-answer pair.

        Args:
            question (str): The input question.
            answer (str): The corresponding answer.

        Returns:
            list[float]: The relevance score.
        """
        try:
            score = self.model.predict([[question, answer]]).tolist()
            logger.debug(f"Single score computed: {score}")
            return score
        except Exception as e:
            logger.error(f"Error during single score computation: {e}")
            raise

    def score_batch(self, question: str, answers: list[str]) -> list[float]:
        """
        Compute relevance scores for a batch of question-answer pairs.

        Args:
            question (str): The input question.
            answers (list[str]): A list of corresponding answers.

        Returns:
            list[float]: A list of relevance scores.
        """
        try:
            qa_comb = [[question, answer] for answer in answers]
            scores = self.model.predict(qa_comb).tolist()
            logger.debug(f"Batch scores computed: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error during batch score computation: {e}")
            raise


class Reranker:
    """
    Class for reranking search results based on relevance scores.
    """

    def __init__(self, relevance_evaluator: AbstractRelevanceEvaluator):
        """
        Initialize the Reranker with a relevance evaluator.

        Args:
            relevance_evaluator (AbstractRelevanceEvaluator): An evaluator for computing relevance scores.
        """
        self.relevance_evaluator = relevance_evaluator

    def rerank(
        self,
        query: str,
        search_results: list[Document],
        search_result_text_key: str = "enhanced_text",
    ) -> list[Document]:
        """
        Rerank search results based on relevance scores.

        Args:
            query (str): The input query.
            search_results (list[Document]): A list of search results to be reranked.
            search_result_text_key (str): The key used to extract text for scoring.

        Returns:
            list[Document]: A list of reranked search results.

        Raises:
            ValueError: If the specified text key is not valid for any Document, BaseDocument, or BaseDocument.metadata.
        """
        try:
            # Extract texts for scoring
            searched_texts = []
            for doc in search_results:
                if hasattr(doc, search_result_text_key):
                    # Get the attribute directly from Document
                    searched_texts.append(getattr(doc, search_result_text_key))
                elif hasattr(doc.base_document, search_result_text_key):
                    # Get the attribute from BaseDocument
                    searched_texts.append(
                        getattr(doc.base_document, search_result_text_key)
                    )
                elif search_result_text_key in doc.base_document.metadata:
                    # Get the value from BaseDocument.metadata dictionary
                    searched_texts.append(
                        doc.base_document.metadata[search_result_text_key]
                    )
                else:
                    raise ValueError(
                        f"{search_result_text_key} is not a valid attribute of Document, "
                        f"BaseDocument, or a key in BaseDocument.metadata."
                    )

            # Compute scores
            scores = self.relevance_evaluator.score_batch(
                question=query, answers=searched_texts
            )

            # Combine results with scores
            reranked_results = [
                Document(
                    base_document=doc.base_document,
                    enhanced_text=doc.enhanced_text,
                    rerank=score,  # Add the score
                )
                for doc, score in zip(search_results, scores)
            ]

            # Sort by score in descending order
            reranked_results.sort(key=lambda x: x.rerank, reverse=True)
            logger.info(f"Reranking completed for query: {query}")
            return reranked_results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise


# %%
