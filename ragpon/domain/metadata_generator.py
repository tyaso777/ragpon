# %%
from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)

TMetadata = TypeVar("TMetadata")


class AbstractMetadataGenerator(ABC, Generic[TMetadata]):
    """
    Abstract base class for metadata generators.

    This class defines the interface for generating metadata objects.
    """

    def __init__(self, metadata_class: Type[TMetadata]):
        """
        Initializes the metadata generator with the given metadata class.

        Args:
            metadata_class (Type[TMetadata]): The class used to create metadata instances.
        """
        self.metadata_class = metadata_class
        logger.debug(
            "AbstractMetadataGenerator initialized with metadata_class: %s",
            metadata_class,
        )

    @abstractmethod
    def generate(self, **kwargs: Any) -> TMetadata:
        """
        Abstract method to generate a metadata instance.

        Args:
            kwargs: Arbitrary key-value pairs needed to create metadata.

        Returns:
            TMetadata: An instance of the metadata class.
        """
        pass


class CustomMetadataGenerator(AbstractMetadataGenerator[TMetadata]):
    """
    Concrete implementation of AbstractMetadataGenerator for generating metadata.
    """

    def __init__(self, metadata_class: Type[TMetadata]):
        """
        Initializes the custom metadata generator with the given metadata class.

        Args:
            metadata_class (Type[TMetadata]): The class used to create metadata instances.
        """
        # (A) Raise TypeError if metadata_class is not callable
        if not callable(metadata_class):
            raise TypeError(f"{metadata_class!r} is not callable as a class.")

        super().__init__(metadata_class)
        logger.info(
            "CustomMetadataGenerator initialized with metadata_class: %s",
            metadata_class,
        )

    def generate(self, **kwargs: Any) -> TMetadata:
        """
        Generates metadata while ensuring compatibility with the user-defined metadata structure.

        Args:
            kwargs: A dictionary of key-value pairs required for metadata. Must include `doc_id` or `id`.

        Returns:
            TMetadata: An instance of the metadata class with mapped fields.

        Raises:
            ValueError: If the required `doc_id` field is missing.
        """
        logger.debug("Generating metadata with input kwargs: %s", kwargs)

        # Convert 'id' to 'doc_id' for consistency with user-defined metadata
        if "id" in kwargs:
            kwargs["doc_id"] = kwargs.pop("id")
            logger.debug("Converted 'id' to 'doc_id' in kwargs")

        if "doc_id" not in kwargs:
            logger.error("Missing required parameter: doc_id")
            raise ValueError("Missing required parameter: doc_id")

        # Extract valid parameters for metadata_class
        valid_params = self.metadata_class.__init__.__code__.co_varnames

        # Separate fields for direct initialization vs metadata
        direct_fields = {
            key: value
            for key, value in kwargs.items()
            if key in valid_params and key != "self"
        }
        metadata_fields = {
            key: value for key, value in kwargs.items() if key not in valid_params
        }

        # If the metadata_class has a 'metadata' field, populate it with remaining kwargs
        if "metadata" in valid_params:
            direct_fields["metadata"] = metadata_fields

        logger.debug("Direct fields for metadata creation: %s", direct_fields)
        logger.debug("Metadata fields: %s", metadata_fields)

        # Create the metadata instance
        metadata = self.metadata_class(**direct_fields)
        logger.info("Metadata instance generated: %s", metadata)
        return metadata
