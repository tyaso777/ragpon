# %%
import json
import os
from pathlib import Path

import yaml

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


class Config:
    """
    Configuration manager that prioritizes values in the following order:
    1. Environment variables
    2. User-defined configuration file (YAML or JSON format) or provided dictionary
    3. Default values
    """

    # Default values
    DEFAULTS = {
        "MODELS": {
            "MULTILINGUAL_ES_LARGE_MODEL_PATH": "/default/path/to/multilingual-e5-large",
            "CL_NAGOYA_RURI_LARGE_MODEL_PATH": "/default/path/to/cl-nagoya-ruri-large",
            "JAPANESE_RERANKER_CROSS_ENCODER_LARGE_V1_PATH": "/default/path/to/japanese-reranker-cross-encoder-large-v1",
            "RURI_RERANKER_LARGE_PATH": "/default/path/to/ruri-reranker-large",
            "MIXTRAL8X7BInstV01_MODEL_PATH": "/default/path/to/mixtral-8x7b-v01",
            "OTHER_MODEL_PATH": "/default/path/to/other-model",
            "TEST_MODEL_PATH": "/default/path/to/test-model",
        },
        "DATABASES": {
            "USE_BM25": "True",  # Enable or disable BM25
            "BM25_PATH": "None",  # Use "None" for in-memory or provide a file path
            "USE_CHROMADB": "True",  # Enable or disable ChromaDB
            "CHROMADB_COLLECTION_NAME": "default_collection",  # ChromaDB collection name
            "CHROMADB_FOLDER_PATH": "None",  # Use "None" for in-memory or provide a folder path
        },
    }

    def __init__(
        self, config_file: str | Path | dict | None = None, encoding: str = "utf-8"
    ):
        """
        Initialize the Config manager.

        Args:
            config_file (str | Path | dict | None, optional):
                Optional path or Path object to a configuration file (YAML or JSON) or a dict containing configuration.
                Defaults to None, which leads to ~/.my_library_config.yaml.
            encoding (str, optional):
                Encoding to use when reading/writing the config file.
                Defaults to "utf-8".
        """
        self.encoding = encoding

        if isinstance(config_file, dict):
            self.config = config_file
            self.config_file = None
            logger.info("Initialized Config from provided dictionary.")
        else:
            self.config_file = (
                Path(config_file)
                if config_file
                else Path.home() / ".my_library_config.yaml"
            )
            logger.info(f"Initializing Config with config file: {self.config_file}")
            self.config = self._load_config_file()

    def _load_config_file(self) -> dict:
        """
        Load configuration from a YAML or JSON file if it exists.

        Returns:
            dict: Configuration dictionary from the file, or an empty dictionary if the file doesn't exist.
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding=self.encoding) as f:
                    if self.config_file.suffix in {".yaml", ".yml"}:
                        # Parse YAML file
                        config = yaml.safe_load(f)
                    elif self.config_file.suffix == ".json":
                        # Parse JSON file
                        config = json.load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file format: {self.config_file.suffix}"
                        )
                    logger.info(f"Loaded configuration file: {self.config_file}")
                    return config
            except (yaml.YAMLError, json.JSONDecodeError, IOError) as e:
                logger.error(
                    f"Failed to load configuration file {self.config_file}: {e}"
                )
                raise RuntimeError(f"Configuration file error: {e}")
        logger.warning(f"Configuration file not found: {self.config_file}")
        return {}

    def get(self, key: str, default: str | None = None) -> str | None:
        """
        Retrieve a configuration value with the following priority:
          1) Environment variables
          2) self.config (loaded from file or provided dict)
          3) DEFAULTS
          4) default argument

        Args:
            key (str): The configuration key to retrieve (e.g., "MODELS.MULTILINGUAL_ES_LARGE_MODEL_PATH").
            default (str, optional):
                Default value to return if the key is not found. Defaults to None.

        Returns:
            str | None: The configuration value if found, otherwise default.
        """
        # 1) Environment variable override
        if key in os.environ:
            return os.environ[key]

        # 2) Check user config
        user_value = self._get_nested(self.config, key.split("."))
        if user_value is not None:
            return user_value

        # 3) Check DEFAULTS
        default_value = self._get_nested(self.DEFAULTS, key.split("."))
        if default_value is not None:
            return default_value

        # 4) Fallback to the default argument
        logger.warning(f"Key '{key}' not found. Using default value: {default}")
        return default

    def _get_nested(
        self, dictionary: dict, keys: list[str], default: str | None = None
    ) -> str | None:
        """
        Safely get a nested value from a dict using a list of keys.
        Returns None if any nested level is missing.

        Args:
            dictionary (dict): The dictionary to navigate.
            keys (list[str]): A list of keys to use for nested lookup.
            default (str | None, optional): Value to return if a key is not found.
                                            Defaults to None.

        Returns:
            str | None: The nested value if found and is a string, otherwise default.
        """
        current = dictionary
        for k in keys:
            if not isinstance(current, dict) or (k not in current):
                return default
            current = current[k]
        return current if isinstance(current, str) else default

    def set(self, key: str, value: str) -> None:
        """
        Set a configuration value in the user-defined configuration dictionary.

        Args:
            key (str): The configuration key to set (e.g., "MODELS.TEST_MODEL_PATH").
            value (str): The value to associate with the key.
        """
        keys = key.split(".")
        config_section = self.config
        for k in keys[:-1]:
            config_section = config_section.setdefault(k, {})
        config_section[keys[-1]] = value
        logger.info(f"Set configuration key '{key}' to value: {value}")

    def save(self, path: str | Path | None = None) -> None:
        """
        Save the current user-defined configuration to the configuration file.
        Optionally, an alternative path can be provided.

        If the configuration was initialized from a dict (i.e. no file path), and no path is provided,
        saving is skipped.

        Args:
            path (str | Path | None, optional): An alternative file path for saving the configuration.
                Defaults to None.
        """
        target_path = Path(path) if path is not None else self.config_file

        if target_path is None:
            logger.warning(
                "No config file path provided. This configuration is in-memory only and will not be saved."
            )
            return
        try:
            with open(target_path, "w", encoding=self.encoding) as f:
                if target_path.suffix in {".yaml", ".yml"}:
                    yaml.dump(self.config, f, default_flow_style=False)
                elif target_path.suffix == ".json":
                    json.dump(self.config, f, indent=4)
                else:
                    raise ValueError("Unsupported config file format.")
            logger.info(f"Configuration saved to: {target_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration file {target_path}: {e}")
            raise RuntimeError(f"Error saving configuration file: {e}")
