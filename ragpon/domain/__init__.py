# ragpon/domain/__init__.py

import importlib
from typing import Any

_lazy_imports = {
    # document_processing_pipeline
    "AbstractDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    "DataFrameDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    "FilePathDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    # document_reader
    "AbstractDocumentReader": "ragpon.domain.document_reader",
    "AbstractDocumentReaderFactory": "ragpon.domain.document_reader",
    "ExtensionBasedDocumentReaderFactory": "ragpon.domain.document_reader",
    # domain
    "BaseDocument": "ragpon.domain.domain",
    "ChunkSourceInfo": "ragpon.domain.domain",
    "Document": "ragpon.domain.domain",
    # metadata_generator
    "AbstractMetadataGenerator": "ragpon.domain.metadata_generator",
    "CustomMetadataGenerator": "ragpon.domain.metadata_generator",
}


def __getattr__(name: str) -> Any:
    """Dynamically import attributes listed in _lazy_imports only when accessed."""
    if name in _lazy_imports:
        module = importlib.import_module(_lazy_imports[name])
        value = getattr(module, name)
        globals()[name] = value  # cache to avoid repeated import
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
