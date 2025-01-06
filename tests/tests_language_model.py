from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from ragpon.config.config import Config
from ragpon.ml_models.large_language_model import Mixtral8x7BInstructV01


@pytest.fixture
def test_config_file():
    """
    Returns the path to a pre-existing static configuration JSON file
    located in the test folder.
    """
    return "tests/test_config.json"  # Adjust path to your test JSON file


def test_mixtral8x7b_with_real_config(test_config_file):
    """
    Test that Mixtral8x7BInstructV01 works with a real config file and model path.
    """
    # Load the actual configuration from the test_config_file
    config = Config(config_file=str(test_config_file))

    # Retrieve the model path from the config
    model_path = config.get("MODELS.MIXTRAL8X7BInstV01_MODEL_PATH")
    assert model_path is not None

    # Initialize the evaluator with the actual model path
    evaluator = Mixtral8x7BInstructV01(config)

    # Use the model to predict on a sample input
    input_text = "これはテスト入力です。"  # Example Japanese input text

    # Mock the model's generate function
    with patch.object(
        evaluator.model, "generate", return_value=np.array([[0]])
    ) as mock_generate:
        with patch.object(
            evaluator.tokenizer, "decode", return_value="これは生成されたテキストです。"
        ) as mock_decode:
            result = evaluator.generate(input_text)

            # Validate the result
            mock_generate.assert_called_once()
            mock_decode.assert_called_once()
            assert isinstance(result, str), "The result should be a string."
            assert len(result) > 0, "The result should not be empty."
            assert (
                "生成" in result
            ), "The result should include '生成' to indicate generation."


@patch("mixtral8x7b_logger.AutoTokenizer.from_pretrained")
@patch("mixtral8x7b_logger.AutoModelForCausalLM.from_pretrained")
def test_mixtral8x7b_initialization(mock_model, mock_tokenizer, test_config_file):
    """
    Test that Mixtral8x7BInstructV01 initializes correctly with a valid config.
    """
    config = Config(config_file=str(test_config_file))

    mock_tokenizer.return_value = MagicMock()
    mock_model.return_value = MagicMock()

    evaluator = Mixtral8x7BInstructV01(config)

    mock_tokenizer.assert_called_once_with(
        config.get("MODELS.MIXTRAL8X7BInstV01_MODEL_PATH"), use_safetensors=True
    )
    mock_model.assert_called_once_with(
        config.get("MODELS.MIXTRAL8X7BInstV01_MODEL_PATH"),
        torch_dtype=np.float16,  # Adjusted for bfloat16 as needed
        quantization_config=ANY,
        device_map="auto",
        trust_remote_code=False,
        use_safetensors=True,
        pad_token_id=ANY,
    )
    assert evaluator is not None
