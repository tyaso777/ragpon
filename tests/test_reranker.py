from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from ragpon._utils.logging_helper import get_library_logger
from ragpon.config.config import Config
from ragpon.domain.domain import BaseDocument, Document, EnhancedInfo, RerankInfo
from ragpon.ml_models.reranker import (
    AbstractRelevanceEvaluator,
    JapaneseRerankerCrossEncoderLargeV1Evaluator,
    Reranker,
)

logger = get_library_logger(__name__)


@pytest.fixture
def config_mock(mocker):
    """
    Returns a Config mock with a valid model path by default.
    """
    cfg = Config()
    mocker.patch.object(cfg, "get", return_value="fake/model/path")  # Any string
    return cfg


@pytest.fixture
def cross_encoder_mock(mocker):
    """
    Fully mock the CrossEncoder constructor and instance so no real call to HF Hub is made.
    Returns (init_mock, instance_mock).
    """
    # Mock the constructor so it does nothing
    init_mock = mocker.patch(
        "sentence_transformers.CrossEncoder.__init__", return_value=None
    )
    # Mock instance
    instance_mock = MagicMock()
    # Patch the predict method at the class level to forward to instance_mock
    mocker.patch(
        "sentence_transformers.CrossEncoder.predict", new=instance_mock.predict
    )
    return init_mock, instance_mock


def test_japanese_reranker_init_success(config_mock, cross_encoder_mock):
    """
    Test that the evaluator initializes correctly with a valid model path.
    """
    init_mock, instance_mock = cross_encoder_mock

    evaluator = JapaneseRerankerCrossEncoderLargeV1Evaluator(config_mock)
    init_mock.assert_called_once_with("fake/model/path", max_length=512, device=ANY)
    assert evaluator is not None


def test_japanese_reranker_init_no_path(mocker):
    """
    Test that the evaluator raises ValueError if model path is missing in config.
    """
    config_without_path = Config()
    mocker.patch.object(config_without_path, "get", return_value=None)

    with pytest.raises(ValueError, match="Model path for Japanese Reranker is not set"):
        _ = JapaneseRerankerCrossEncoderLargeV1Evaluator(config_without_path)


def test_japanese_reranker_score_single_numpy_array(config_mock, cross_encoder_mock):
    """
    Test that score_single returns a numpy array (even if size == 1).
    """
    init_mock, instance_mock = cross_encoder_mock
    single_score = np.array([0.1234], dtype=np.float32)
    instance_mock.predict.return_value = single_score

    evaluator = JapaneseRerankerCrossEncoderLargeV1Evaluator(config_mock)
    result = evaluator.score_single("question", "answer")

    # Verify result is a NumPy array and close to 0.1234
    assert isinstance(result, np.ndarray)
    assert np.isclose(result, 0.1234, rtol=1e-5)

    instance_mock.predict.assert_called_once_with([("question", "answer")])


def test_japanese_reranker_score_batch(config_mock, cross_encoder_mock):
    """
    Test score_batch returns a NumPy array of floats from CrossEncoder.
    """
    init_mock, instance_mock = cross_encoder_mock
    batch_scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    instance_mock.predict.return_value = batch_scores

    evaluator = JapaneseRerankerCrossEncoderLargeV1Evaluator(config_mock)
    answers = ["ans1", "ans2", "ans3"]
    result = evaluator.score_batch("query", answers)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, batch_scores)

    instance_mock.predict.assert_called_once_with(
        [
            ("query", "ans1"),
            ("query", "ans2"),
            ("query", "ans3"),
        ]
    )


def test_reranker_happy_path():
    """
    Test that Reranker extracts text, scores them, and sorts documents by descending score.
    """
    evaluator_mock = MagicMock(spec=AbstractRelevanceEvaluator)
    evaluator_mock.score_batch.return_value = np.array(
        [0.9, 0.2, 0.7], dtype=np.float32
    )

    # Provide a minimal BaseDocument so doc_id is not None
    doc1 = Document(
        base_document=BaseDocument(doc_id="doc1", text=""),
        enhanced=EnhancedInfo(enhanced_text="Doc1 text"),
        rerank=None,
    )
    doc2 = Document(
        base_document=BaseDocument(doc_id="doc2", text=""),
        enhanced=EnhancedInfo(enhanced_text="Doc2 text"),
        rerank=None,
    )
    doc3 = Document(
        base_document=BaseDocument(doc_id="doc3", text=""),
        enhanced=EnhancedInfo(enhanced_text="Doc3 text"),
        rerank=None,
    )
    documents = [doc1, doc2, doc3]

    reranker = Reranker(evaluator_mock)
    results = reranker.rerank(
        "my query", documents, search_result_text_key="enhanced_text"
    )

    # Scores: doc1(0.9), doc3(0.7), doc2(0.2)
    assert results[0].base_document.doc_id == "doc1"
    # Compare with a small tolerance
    assert np.isclose(results[0].rerank.score, 0.9, rtol=1e-5)

    assert results[1].base_document.doc_id == "doc3"
    assert np.isclose(results[1].rerank.score, 0.7, rtol=1e-5)

    assert results[2].base_document.doc_id == "doc2"
    assert np.isclose(results[2].rerank.score, 0.2, rtol=1e-5)

    evaluator_mock.score_batch.assert_called_once_with(
        question="my query",
        answers=["Doc1 text", "Doc2 text", "Doc3 text"],
    )


def test_reranker_invalid_key():
    """
    Test that Reranker raises AssertionError if the specified text key is not in EnhancedInfo.
    """
    evaluator_mock = MagicMock(spec=AbstractRelevanceEvaluator)
    reranker = Reranker(evaluator_mock)

    doc = Document(
        base_document=BaseDocument(doc_id="docX", text=""),
        enhanced=EnhancedInfo(enhanced_text="some text"),
        rerank=None,
    )

    with pytest.raises(AssertionError, match="not a valid key of EnhancedInfo"):
        reranker.rerank("query", [doc], search_result_text_key="nonexistent_key")


@pytest.fixture
def test_config_file():
    """
    Returns the path to a pre-existing static configuration JSON file
    located in the test folder.
    """
    return "tests/test_config.json"  # Adjust path to your test JSON file


def test_japanese_reranker_with_real_config(test_config_file):
    """
    Test that JapaneseRerankerCrossEncoderLargeV1Evaluator works with a real config file and model path.
    """
    # Load the actual configuration from the test_config_file
    config = Config(config_file=str(test_config_file))

    # Retrieve the model path from the config
    model_path = config.get("MODELS.JAPANESE_RERANKER_CROSS_ENCODER_LARGE_V1_PATH")
    assert model_path is not None

    # Initialize the evaluator with the actual model path
    evaluator = JapaneseRerankerCrossEncoderLargeV1Evaluator(config)

    # Use the model to predict on a sample input
    question = "これはテスト質問です。"  # Example Japanese question
    answer = "これはテスト回答です。"  # Example Japanese answer

    # Perform a prediction
    result = evaluator.score_single(question, answer)

    # Validate the result
    assert isinstance(result, np.ndarray), "The result should be a NumPy array."
    assert result.size == 1, "The result should contain exactly one score."
    assert 0.0 <= result[0] <= 1.0, "The score should be between 0 and 1."
