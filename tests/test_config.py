import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ragpon.config.config import Config  # Adjust import path if necessary


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """
    A pytest fixture that provides a temporary config file path.
    """
    return tmp_path / "test_config.json"


def test_load_from_existing_file(temp_config_file: Path):
    """
    Test that the Config object correctly loads values from an existing JSON file.
    """
    sample_data = {
        "MODELS": {"TEST_MODEL_PATH": "/custom/path/to/test-model"},
        "LOGGING": {"LOG_LEVEL": "DEBUG"},
    }
    temp_config_file.write_text(json.dumps(sample_data), encoding="utf-8")

    config = Config(config_file=str(temp_config_file))

    assert config.get("MODELS.TEST_MODEL_PATH") == "/custom/path/to/test-model"
    assert config.get("LOGGING.LOG_LEVEL") == "DEBUG"


def test_load_from_non_existent_file(temp_config_file: Path, caplog):
    """
    Test that the Config object does not raise an error when the file doesn't exist,
    and that it falls back to default values.
    """
    if temp_config_file.exists():
        temp_config_file.unlink()

    with caplog.at_level("WARNING"):
        config = Config(config_file=str(temp_config_file))
        assert config.get("MODELS.TEST_MODEL_PATH") == "/default/path/to/test-model"
        assert config.get("LOGGING.LOG_LEVEL") == "INFO"

    # Verify log messages
    assert "Configuration file not found" in caplog.text
    assert "Key 'MODELS.TEST_MODEL_PATH' not found" not in caplog.text


def test_environment_variable_override(temp_config_file: Path):
    """
    Test that environment variables take precedence over both the config file and defaults.
    """
    sample_data = {
        "MODELS": {"TEST_MODEL_PATH": "/file/path/to/test-model"},
        "LOGGING": {"LOG_LEVEL": "DEBUG"},
    }
    temp_config_file.write_text(json.dumps(sample_data), encoding="utf-8")

    with patch.dict(os.environ, {"MODELS.TEST_MODEL_PATH": "/env/path/to/test-model"}):
        config = Config(config_file=str(temp_config_file))
        assert config.get("MODELS.TEST_MODEL_PATH") == "/env/path/to/test-model"

    config = Config(config_file=str(temp_config_file))
    assert config.get("MODELS.TEST_MODEL_PATH") == "/file/path/to/test-model"
    assert config.get("LOGGING.LOG_LEVEL") == "DEBUG"


def test_set_and_get_value(temp_config_file: Path):
    """
    Test that we can set and retrieve a value in the in-memory config.
    """
    config = Config(config_file=str(temp_config_file))

    config.set("MODELS.TEST_MODEL_PATH", "/new/path/to/test-model")
    assert config.get("MODELS.TEST_MODEL_PATH") == "/new/path/to/test-model"

    assert config.get("LOGGING.LOG_LEVEL") == "INFO"


def test_save_configuration(temp_config_file: Path):
    """
    Test that we can save the user-defined config to a file, and load it back.
    """
    config = Config(config_file=str(temp_config_file))

    config.set("MODELS.TEST_MODEL_PATH", "/save/test-model")
    config.set("LOGGING.LOG_LEVEL", "WARNING")
    config.save()

    saved_data = json.loads(temp_config_file.read_text(encoding="utf-8"))
    assert saved_data["MODELS"]["TEST_MODEL_PATH"] == "/save/test-model"
    assert saved_data["LOGGING"]["LOG_LEVEL"] == "WARNING"

    new_config = Config(config_file=str(temp_config_file))
    assert new_config.get("MODELS.TEST_MODEL_PATH") == "/save/test-model"
    assert new_config.get("LOGGING.LOG_LEVEL") == "WARNING"


@pytest.mark.parametrize(
    "key,expected",
    [
        ("MODELS.TEST_MODEL_PATH", "/default/path/to/test-model"),
        ("LOGGING.LOG_LEVEL", "INFO"),
        ("SOME.MISSING.KEY", "my_override_default"),
    ],
)
def test_default_value_is_used(temp_config_file: Path, key: str, expected: str):
    """
    Test that default values are used when no environment variable or config file value exists.
    """
    if temp_config_file.exists():
        temp_config_file.unlink()

    config = Config(config_file=str(temp_config_file))
    default_value = "my_override_default" if key == "SOME.MISSING.KEY" else None
    assert config.get(key, default=default_value) == expected


def test_japanese_path_support(tmp_path: Path):
    """
    Test that the Config class works with paths containing Japanese characters.
    """
    japanese_path = tmp_path / "テスト設定.json"
    sample_data = {
        "MODELS": {"TEST_MODEL_PATH": "/カスタム/パス/テストモデル"},
        "LOGGING": {"LOG_LEVEL": "DEBUG"},  # 適切なログレベルに修正
    }
    japanese_path.write_text(json.dumps(sample_data), encoding="utf-8")

    # Load the configuration
    config = Config(config_file=str(japanese_path))
    assert config.get("MODELS.TEST_MODEL_PATH") == "/カスタム/パス/テストモデル"
    assert config.get("LOGGING.LOG_LEVEL") == "DEBUG"  # 修正済み

    # Set and save new values
    config.set("MODELS.TEST_MODEL_PATH", "/新しい/パス")
    config.set("LOGGING.LOG_LEVEL", "INFO")  # 別の有効なログレベルに変更
    config.save()

    # Verify saved data
    saved_data = json.loads(japanese_path.read_text(encoding="utf-8"))
    assert saved_data["MODELS"]["TEST_MODEL_PATH"] == "/新しい/パス"
    assert saved_data["LOGGING"]["LOG_LEVEL"] == "INFO"
