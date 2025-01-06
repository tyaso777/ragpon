import math
from typing import Generic, Type, TypeVar

from peewee import Model
from playhouse.shortcuts import model_to_dict

from ragpon._utils.logging_helper import get_library_logger
from ragpon.domain.domain import BaseDocument, Document
from ragpon.repository.bm25.models import DataModels
from ragpon.tokenizer import AbstractTokenizer

# Initialize logger
logger = get_library_logger(__name__)

MetadataType = Type[BaseDocument]
ModelType = Type[Model]

TMetadata = TypeVar("TMetadata", bound=BaseDocument)
TResult = TypeVar("TResult", bound=Document)


# %%
class BM25PlusCalculator(Generic[TMetadata, TResult]):
    def __init__(
        self,
        data_models: DataModels,
        tokenizer: AbstractTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ):
        logger.debug(
            "Initializing BM25PlusCalculator with k1=%s, b=%s, delta=%s", k1, b, delta
        )
        self._k1 = k1
        self._b = b
        self._delta = delta  # BM25+ specific delta
        self._data_models = data_models
        self._tokenizer = tokenizer
        self._N = 0
        self._avgdl = 0

    def _get_stats(self):
        stats = self._data_models.Statistics.get()
        self._N = stats.total_documents
        self._avgdl = stats.total_words / self._N if self._N else 0
        logger.debug("Loaded stats: total_documents=%s, avgdl=%s", self._N, self._avgdl)

    def _idf(self, term):
        logger.debug("Calculating IDF for term: %s", term)
        self._get_stats()
        try:
            term_record = self._data_models.Term.get(
                self._data_models.Term.term == term
            )
            df = term_record.document_frequency
            idf = max(0, math.log((self._N - df + 0.5) / (df + 0.5) + 1))
            logger.debug("Computed IDF for term '%s': %s", term, idf)
            return idf
        except self._data_models.Term.DoesNotExist:
            logger.warning("Term '%s' does not exist in the database", term)
            return 0

    def _calculate_scores(self, query: str):
        logger.debug("Calculating scores for query: %s", query)
        self._get_stats()
        scores: dict[str, float] = {}
        query_terms = self._tokenizer.tokenize(query)
        logger.debug("Tokenized query terms: %s", query_terms)

        for term in query_terms:
            idf = self._idf(term)
            for tf in (
                self._data_models.TermFrequency.select(
                    self._data_models.TermFrequency, self._data_models.Document
                )
                .join(
                    self._data_models.Document,
                    on=(
                        self._data_models.TermFrequency.document_id
                        == self._data_models.Document.id
                    ),
                    attr="document",
                )
                .where(self._data_models.TermFrequency.term == term)
            ):
                doc_id = tf.document_id
                tf_value = tf.term_frequency
                doc_length = tf.document.length
                numerator = tf_value * (self._k1 + 1)
                denominator = tf_value + self._k1 * (
                    1 - self._b + self._b * (doc_length / self._avgdl)
                )
                score = idf * (numerator / denominator + self._delta)
                scores[doc_id] = scores.get(doc_id, 0) + score
                logger.debug(
                    "Updated score for doc_id '%s': %s", doc_id, scores[doc_id]
                )

        return scores

    def search(self, query: str, top_k: int) -> list[TResult]:
        logger.info("Searching for query: %s with top_k=%d", query, top_k)
        scores = self._calculate_scores(query)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :top_k
        ]

        results = []
        for id, score in sorted_scores:
            try:
                metadata_record = self._data_models.Document.get(
                    self._data_models.Document.id == id
                )
                metadata_dict = model_to_dict(metadata_record)
                text = metadata_dict.pop("text", None)
                db_name = metadata_dict.pop("db_name", "bm25")
                distance = metadata_dict.pop("distance", 1)  # value before calculation
                distance = 1.0 / (score + 1)  # calculated value

                base_document = BaseDocument(
                    doc_id=str(id),
                    db_name=db_name,
                    distance=distance,
                    text=text,
                    metadata=metadata_dict["metadata"],
                )

                result = Document(base_document=base_document)
                results.append(result)
                logger.debug("Appended result for doc_id '%s': %s", id, result)
            except self._data_models.Document.DoesNotExist:
                logger.warning("Document with id '%s' does not exist", id)

        logger.info("Search completed with %d results", len(results))
        return results
