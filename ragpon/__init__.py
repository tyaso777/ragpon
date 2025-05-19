"""
Ragpon - Retrieval Augmented Generation for Python
-------------------------------------------------

This package uses a lazy import system to avoid loading all dependencies
at import time. This allows you to use only the modules you need.
"""

import importlib

__all__ = [
    # Config
    "Config",
    "ProxyConfigurator",
    # Chunk processors
    "AbstractChunkProcessor",
    "FixedLengthChunkProcessor",
    "JAGinzaChunkProcessor",
    "NaturalChunkProcessor",
    "SingleChunkProcessor",
    # Domain models
    "BaseDocument",
    "ChunkSourceInfo",
    "Document",
    "CustomMetadataGenerator",
    "AbstractDocumentProcessingPipeline",
    "DataFrameDocumentProcessingPipeline",
    "FilePathDocumentProcessingPipeline",
    "AbstractDocumentReaderFactory",
    "ExtensionBasedDocumentReaderFactory",
    "AbstractMetadataGenerator",
    # ML Models
    "AbstractEmbeddingModel",
    "AbstractLargeLanguageModel",
    "AbstractRelevanceEvaluator",
    "AbstractSimilarityCalculator",
    "ChromaDBEmbeddingAdapter",
    "CosineSimilarityCalculator",
    "JapaneseRerankerCrossEncoderLargeV1Evaluator",
    "Mixtral8x7BInstructV01",
    "MultilingualE5LargeEmbedder",
    "Reranker",
    "RuriLargeEmbedder",
    "RuriLargeEmbedderCTranslate2",
    "RuriRerankerLargeEvaluator",
    # Repositories
    "AbstractRepository",
    "AbstractResultsFormatter",
    "BM25Repository",
    "ChromaDBRepository",
    "ChromaDBResultsFormatter",
    # Services
    "DocumentProcessingService",
]

_lazy_imports: dict[str, str] = {
    # Config
    "Config": "ragpon.config.config",
    "ProxyConfigurator": "ragpon.config.proxy_configurator",
    # Chunk processors
    "AbstractChunkProcessor": "ragpon.chunk_processor",
    "FixedLengthChunkProcessor": "ragpon.chunk_processor",
    "JAGinzaChunkProcessor": "ragpon.chunk_processor",
    "NaturalChunkProcessor": "ragpon.chunk_processor",
    "SingleChunkProcessor": "ragpon.chunk_processor",
    # Domain
    "BaseDocument": "ragpon.domain.domain",
    "ChunkSourceInfo": "ragpon.domain.domain",
    "Document": "ragpon.domain.domain",
    "CustomMetadataGenerator": "ragpon.domain.metadata_generator",
    "AbstractMetadataGenerator": "ragpon.domain.metadata_generator",
    "AbstractDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    "DataFrameDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    "FilePathDocumentProcessingPipeline": "ragpon.domain.document_processing_pipeline",
    "AbstractDocumentReaderFactory": "ragpon.domain.document_reader",
    "ExtensionBasedDocumentReaderFactory": "ragpon.domain.document_reader",
    # ML Models
    "AbstractEmbeddingModel": "ragpon.ml_models.embedding_model",
    "AbstractSimilarityCalculator": "ragpon.ml_models.embedding_model",
    "ChromaDBEmbeddingAdapter": "ragpon.ml_models.embedding_model",
    "CosineSimilarityCalculator": "ragpon.ml_models.embedding_model",
    "MultilingualE5LargeEmbedder": "ragpon.ml_models.embedding_model",
    "RuriLargeEmbedder": "ragpon.ml_models.embedding_model",
    "RuriLargeEmbedderCTranslate2": "ragpon.ml_models.embedding_model",
    "AbstractLargeLanguageModel": "ragpon.ml_models.large_language_model",
    "Mixtral8x7BInstructV01": "ragpon.ml_models.large_language_model",
    "AbstractRelevanceEvaluator": "ragpon.ml_models.reranker",
    "JapaneseRerankerCrossEncoderLargeV1Evaluator": "ragpon.ml_models.reranker",
    "Reranker": "ragpon.ml_models.reranker",
    "RuriRerankerLargeEvaluator": "ragpon.ml_models.reranker",
    # Repository
    "AbstractRepository": "ragpon.repository.abstract_repository",
    "BM25Repository": "ragpon.repository.bm25",
    "ChromaDBRepository": "ragpon.repository.chromaDB_repository",
    "AbstractResultsFormatter": "ragpon.repository.search_results_formatter",
    "ChromaDBResultsFormatter": "ragpon.repository.search_results_formatter",
    # Service
    "DocumentProcessingService": "ragpon.service.document_processing_service",
}


def __getattr__(name: str):
    """
    Lazily import a module attribute on first access.

    Args:
        name (str): Name of the attribute to import.

    Returns:
        Any: The imported symbol.

    Raises:
        AttributeError: If the attribute is not found in the lazy import map.
    """
    if name in _lazy_imports:
        module_path = _lazy_imports[name]
        module = importlib.import_module(module_path)
        try:
            return getattr(module, name)
        except AttributeError as e:
            raise ImportError(f"Module '{module_path}' does not define '{name}'") from e
    raise AttributeError(f"module 'ragpon' has no attribute '{name}'")
