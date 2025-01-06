import time

import pytest

from ragpon.domain.domain import (
    BaseDocument,
    ChunkSourceInfo,
    Document,
    EnhancedInfo,
    RerankInfo,
)


def test_chunk_source_info_init(caplog):
    """
    Test that ChunkSourceInfo initializes correctly
    and logs a debug message about unit_index and text length.
    """
    with caplog.at_level("DEBUG"):
        chunk = ChunkSourceInfo(
            text="Example chunk", unit_index=42, metadata={"key": "value"}
        )

    assert chunk.text == "Example chunk"
    assert chunk.unit_index == 42
    assert chunk.metadata["key"] == "value"

    # Verify the log contains the correct initialization message
    assert (
        "Initialized ChunkSourceInfo with unit_index: 42, text length: 13"
        in caplog.text
    )


def test_base_document_init(caplog):
    """
    Test that BaseDocument initializes correctly
    and logs a debug message about doc_id and text length.
    """
    with caplog.at_level("DEBUG"):
        doc = BaseDocument(doc_id="test_doc", text="Hello World")

    assert doc.doc_id == "test_doc"
    assert doc.text == "Hello World"
    assert doc.db_name is None
    assert doc.distance is None
    assert doc.metadata == {}

    # Verify the log contains the correct initialization message
    assert (
        "Initialized BaseDocument with doc_id: test_doc, text length: 11" in caplog.text
    )


def test_enhanced_info_init(caplog):
    """
    Test that EnhancedInfo initializes correctly
    and logs the length of enhanced_text.
    """
    with caplog.at_level("DEBUG"):
        info = EnhancedInfo(enhanced_text="Some extra info")

    assert info.enhanced_text == "Some extra info"

    # Verify the log contains the correct initialization message
    assert "Initialized EnhancedInfo with enhanced_text length: 15" in caplog.text


def test_rerank_info_init(caplog):
    """
    Test that RerankInfo initializes correctly
    and logs the score.
    """
    with caplog.at_level("DEBUG"):
        rerank = RerankInfo(score=9.99)

    assert rerank.score == 9.99

    # Verify the log contains the correct initialization message
    assert "Initialized RerankInfo with score: 9.990000" in caplog.text


def test_document_init_no_enhanced_no_rerank(caplog):
    """
    Test that Document initializes with only a BaseDocument,
    and logs the correct debug message (enhanced_text=none, rerank=none).
    """
    base = BaseDocument(doc_id="doc123", text="Sample text")

    with caplog.at_level("DEBUG"):
        doc = Document(base_document=base)

    assert doc.base_document == base
    assert doc.enhanced_text is None
    assert doc.rerank is None

    # Verify the log contains the correct initialization message
    assert (
        "Initialized Document with doc_id: doc123, enhanced_text: none, rerank: none"
        in caplog.text
    )


def test_document_init_with_enhanced_and_rerank(caplog):
    """
    Test that Document initializes with EnhancedInfo and RerankInfo,
    and logs the correct debug message (enhanced_text=present, rerank=present).
    """
    base = BaseDocument(doc_id="docABC", text="Test text")
    enhanced_text = EnhancedInfo(enhanced_text="enhanced_text sample")
    rerank = RerankInfo(score=42.0)

    with caplog.at_level("DEBUG"):
        doc = Document(base_document=base, enhanced=enhanced, rerank=rerank)

    assert doc.base_document == base
    assert doc.enhanced == enhanced
    assert doc.rerank == rerank

    # Verify the log contains the correct initialization message
    assert (
        "Initialized Document with doc_id: docABC, enhanced: present, rerank: present"
        in caplog.text
    )


def test_base_document_metadata(caplog):
    """
    Test that BaseDocument can store arbitrary metadata
    and logs initialization properly.
    """
    with caplog.at_level("DEBUG"):
        base = BaseDocument(
            doc_id="meta_doc",
            text="Metadata test",
            db_name="test_db",
            distance=0.123,
            metadata={"key1": "value1", "page_number": 5},
        )

        assert base.doc_id == "meta_doc"
        assert base.text == "Metadata test"
        assert base.db_name == "test_db"
        assert base.distance == 0.123
        assert base.metadata["key1"] == "value1"
        assert base.metadata["page_number"] == 5

        # Verify the log contains the correct initialization message
        assert (
            "Initialized BaseDocument with doc_id: meta_doc, text length: 13"
            in caplog.text
        )


def test_base_document_invalid_text_type():
    """
    Test that BaseDocument raises a TypeError when text is not a string.
    """
    with pytest.raises(TypeError):
        BaseDocument(doc_id="invalid_doc", text=12345)  # text should be a string


def test_base_document_missing_doc_id():
    """
    Test that BaseDocument raises a TypeError when doc_id is missing.
    """
    with pytest.raises(TypeError):
        BaseDocument(text="Missing doc_id")  # doc_id is required


@pytest.mark.parametrize(
    "doc_id, text, db_name, distance, metadata",
    [
        ("doc1", "Sample text 1", None, None, {}),
        ("doc2", "Sample text 2", "db2", 0.5, {"key": "value"}),
        ("doc3", "Sample text 3", "db3", 1.23, {"page": 10}),
    ],
)
def test_base_document_parametrize(doc_id, text, db_name, distance, metadata, caplog):
    """
    Test BaseDocument initialization with multiple parameter sets.
    """
    with caplog.at_level("DEBUG"):
        doc = BaseDocument(
            doc_id=doc_id,
            text=text,
            db_name=db_name,
            distance=distance,
            metadata=metadata,
        )

    assert doc.doc_id == doc_id
    assert doc.text == text
    assert doc.db_name == db_name
    assert doc.distance == distance
    assert doc.metadata == metadata

    # Verify the log contains the correct initialization message
    assert (
        f"Initialized BaseDocument with doc_id: {doc_id}, text length: {len(text)}"
        in caplog.text
    )


def test_base_document_performance():
    """
    Test the performance of BaseDocument initialization with large text.
    """
    large_text = "a" * 10**6  # 1MB of text
    iterations = 100

    start_time = time.perf_counter()
    doc = BaseDocument(doc_id="large_doc", text=large_text)

    for _ in range(iterations):
        doc = BaseDocument(doc_id="large_doc", text=large_text)
        assert doc.text == large_text  # Verify correctness in each iteration

    end_time = time.perf_counter()

    assert doc.text == large_text
    elapsed_time = end_time - start_time
    assert elapsed_time < 0.001, f"Initialization took too long ({elapsed_time:.6f}s)"
