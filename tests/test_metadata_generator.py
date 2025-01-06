# test_metadata_generator.py

from unittest.mock import patch

import pytest

from ragpon.domain.metadata_generator import CustomMetadataGenerator


# A dummy class to simulate the user-defined "metadata_class"
class DummyMeta:
    def __init__(self, doc_id, text="default text", extra=None):
        self.doc_id = doc_id
        self.text = text
        self.extra = extra

    def __repr__(self):
        return f"DummyMeta(doc_id={self.doc_id!r}, text={self.text!r}, extra={self.extra!r})"


def test_custom_metadata_generator_init(caplog):
    """
    Test that CustomMetadataGenerator initializes correctly and logs at INFO level.
    """
    with caplog.at_level("DEBUG"):
        gen = CustomMetadataGenerator(DummyMeta)

    # Check that the 'AbstractMetadataGenerator' debug log is present
    assert "AbstractMetadataGenerator initialized with metadata_class:" in caplog.text
    # Check that the CustomMetadataGenerator info log is present
    assert "CustomMetadataGenerator initialized with metadata_class:" in caplog.text


def test_generate_with_doc_id(caplog):
    """
    Test that generate() creates an instance of metadata_class when doc_id is provided.
    """
    gen = CustomMetadataGenerator(DummyMeta)
    with caplog.at_level("DEBUG"):
        meta = gen.generate(doc_id="doc123", text="some text", extra=42)

    # Check the resulting object
    assert isinstance(meta, DummyMeta)
    assert meta.doc_id == "doc123"
    assert meta.text == "some text"
    assert meta.extra == 42

    # Verify logs
    assert "Generating metadata with input kwargs:" in caplog.text
    assert "Metadata instance generated:" in caplog.text
    assert "doc_id='doc123'" in caplog.text  # From __repr__ or logs


def test_generate_with_id_conversion(caplog):
    """
    Test that 'id' is converted to 'doc_id' if present in kwargs.
    """
    gen = CustomMetadataGenerator(DummyMeta)
    with caplog.at_level("DEBUG"):
        meta = gen.generate(id="docABC", text="converted test")

    # Check the resulting object
    assert meta.doc_id == "docABC"
    assert meta.text == "converted test"

    # Verify logs
    assert "Converted 'id' to 'doc_id' in kwargs" in caplog.text


def test_generate_missing_doc_id(caplog):
    """
    Test that generate() raises ValueError if doc_id is missing.
    """
    gen = CustomMetadataGenerator(DummyMeta)
    with caplog.at_level("DEBUG"), pytest.raises(
        ValueError, match="Missing required parameter: doc_id"
    ):
        gen.generate(text="no doc id")

    # Check the error log
    assert "Missing required parameter: doc_id" in caplog.text


def test_invalid_metadata_class():
    """
    Test that initializing CustomMetadataGenerator with an invalid class raises TypeError.
    """
    with pytest.raises(TypeError, match="is not callable"):
        CustomMetadataGenerator(None)  # None is not callable


def test_generate_with_unexpected_kwargs(caplog):
    """
    Test that generate() logs a warning for unexpected kwargs.
    """

    class AnotherDummyMeta:
        def __init__(self, doc_id, text="blah"):
            self.doc_id = doc_id
            self.text = text

        def __repr__(self):
            return f"AnotherDummyMeta(doc_id={self.doc_id!r}, text={self.text!r})"

    gen = CustomMetadataGenerator(AnotherDummyMeta)
    with caplog.at_level("WARNING"):
        meta = gen.generate(doc_id="doc_unexpected", unknown="unexpected")

    # Ensure that the metadata was generated
    assert meta.doc_id == "doc_unexpected"
    assert meta.text == "blah"  # default
    # Verify warning logs
    assert "Unexpected keyword argument: 'unknown'" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"doc_id": "d1", "text": "Test text", "extra": "Hello"},
        {"id": "d2", "extra": 123},  # 'id' should be converted to doc_id
        {"doc_id": "d3"},  # No 'text' or 'extra' => defaults
    ],
)
def test_generate_parametrized(kwargs, caplog):
    """
    Parametrized test for different kinds of kwargs.
    """
    gen = CustomMetadataGenerator(DummyMeta)
    with caplog.at_level("DEBUG"):
        meta = gen.generate(**kwargs)

    # If 'id' was given, it should be converted to 'doc_id'
    expected_doc_id = kwargs.get("doc_id") or kwargs.get("id")
    assert meta.doc_id == expected_doc_id

    # Logs check
    assert "Generating metadata with input kwargs:" in caplog.text
    assert "Metadata instance generated:" in caplog.text
