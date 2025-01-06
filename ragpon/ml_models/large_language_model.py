# %%
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ragpon._utils.logging_helper import get_library_logger
from ragpon.config import Config

# Initialize logger
logger = get_library_logger(__name__)


class AbstractLargeLanguageModel(ABC):
    """
    Abstract base class for large language models.
    """

    @abstractmethod
    def __init__(self, config: Config):
        """
        Initialize the large language model with a configuration.

        Args:
            config (Config): Configuration object to load model settings.
        """
        pass

    @abstractmethod
    def generate(self, text: str) -> str:
        """
        Generate a response based on the input text.

        Args:
            text (str): The input text to generate a response.

        Returns:
            str: The generated response.
        """
        pass


class Mixtral8x7BInstructV01(AbstractLargeLanguageModel):
    def __init__(self, config: Config):
        """
        Initializes the Mixtral8x7BInstructV01 with tokenizer and model using a given config.

        Args:
            config (Config): Configuration object to load model paths.
        """
        try:
            model_path = config.get("MODELS.MIXTRAL8X7BInstV01_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "Model path for MIXTRAL8x7BInstV01 is not set in the configuration."
                )

            logger.info(
                "Initializing Mixtral8x7BInstructV01 with model path: %s", model_path
            )

            self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)

            # Initialize the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_safetensors=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=self.quantization_config,
                device_map="auto",
                trust_remote_code=False,
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Load the generation configuration
            self.model_generate_config = {
                "temperature": 0.9,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 1000,  # Adjust for generation diversity
                "max_new_tokens": 256,
            }

            logger.info("Mixtral8x7BInstructV01 initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Mixtral8x7BInstructV01: %s", str(e))
            raise

    def generate(self, text: str) -> str:
        """
        Generates a response based on the input text using the model.

        Args:
            text (str): The input text to generate a response.

        Returns:
            str: The generated response.
        """
        try:
            logger.info("Generating response for input text: %s", text)
            messages = [{"role": "user", "content": text}]

            with torch.no_grad():
                token_ids = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt"
                )
                output_ids = self.model.generate(
                    token_ids.to(self.model.device),
                    pad_token_id=self.tokenizer.eos_token_id,
                    **self.model_generate_config,
                )

            output = self.tokenizer.decode(output_ids[0][token_ids.size(1) : -2])
            logger.info("Generated response: %s", output)
            return output
        except Exception as e:
            logger.error("Failed to generate response: %s", str(e))
            raise
