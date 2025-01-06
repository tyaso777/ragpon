import pytest

from ragpon.chunk_processor import (
    ChunkProcessingError,
    FixedLengthChunkProcessor,
    JAGinzaChunkProcessor,
    NaturalChunkProcessor,
    SingleChunkProcessor,
)


def test_fixed_length_chunk_processor():
    processor = FixedLengthChunkProcessor(chunk_size=5)

    # Test normal input
    text = "HelloWorld"
    chunks = processor.process(text)
    assert chunks == ["Hello", "World"]

    # Test empty input
    assert processor.process("") == []

    # Test small input
    assert processor.process("Hi") == ["Hi"]

    # Test Japanese input
    text = "こんにちは世界"
    chunks = processor.process(text)
    assert chunks == ["こんにちは", "世界"]


def test_natural_chunk_processor():
    processor = NaturalChunkProcessor()

    # Test normal input
    text = "Hello world. How are you?"
    chunks = processor.process(text)
    assert chunks == ["Hello world.", " How are you?"]

    # Test empty input
    assert processor.process("") == []

    # Test input without delimiters
    assert processor.process("Hello") == ["Hello"]

    # Test input with multiple delimiters
    text = "Hi! Are you okay? I'm fine."
    chunks = processor.process(text)
    assert chunks == ["Hi!", " Are you okay?", " I'm fine."]

    # Test Japanese input
    text = "こんにちは。元気ですか？はい、元気です！"
    chunks = processor.process(text)
    assert chunks == ["こんにちは。", "元気ですか？", "はい、元気です！"]


def test_single_chunk_processor():
    processor = SingleChunkProcessor()

    # Test normal input
    text = "Hello world"
    chunks = processor.process(text)
    assert chunks == ["Hello world"]

    # Test empty input
    assert processor.process("") == []

    # Test Japanese input
    text = "こんにちは世界"
    chunks = processor.process(text)
    assert chunks == ["こんにちは世界"]


# TODO: Could not create test case for jaginza_chunk_processor_multiple
# def test_jaginza_chunk_processor_multiple(mocker):
#     from ragpon.chunk_processor import JAGinzaChunkProcessor

#     # Mock spacy.load
#     mock_spacy_load = mocker.patch("spacy.load")
#     nlp_mock = mocker.MagicMock(name="NLP_Mock")
#     mock_spacy_load.return_value = nlp_mock

#     # Mock return values based on input text
#     def mock_nlp_call(text):
#         if "Sentence" in text:
#             mock_doc = mocker.MagicMock(name="Doc_English")
#             mock_doc.sents = [
#                 mocker.Mock(text="Sentence 1."),
#                 mocker.Mock(text="Sentence 2."),
#                 mocker.Mock(text="Sentence 3."),
#             ]
#             return mock_doc
#         elif "こんにちは" in text:
#             mock_doc = mocker.MagicMock(name="Doc_Japanese")
#             mock_doc.sents = [
#                 mocker.Mock(text="こんにちは。"),
#                 mocker.Mock(text="元気ですか？"),
#                 mocker.Mock(text="はい、元気です。"),
#             ]
#             return mock_doc
#         else:
#             mock_doc = mocker.MagicMock(name="Doc_Empty")
#             mock_doc.sents = []
#             return mock_doc

#     nlp_mock.return_value = mock_nlp_call

#     processor = JAGinzaChunkProcessor(chunk_size=20)

#     # Test English text
#     text_en = "Sentence 1. Sentence 2. Sentence 3."
#     chunks_en = processor.process(text_en)
#     assert chunks_en == ["Sentence 1. Sentence 2.", " Sentence 3."]

#     # Test empty input
#     chunks_empty = processor.process("")
#     assert chunks_empty == []

#     # Test Japanese text
#     text_jp = "こんにちは。元気ですか？はい、元気です。"
#     chunks_jp = processor.process(text_jp)
#     assert chunks_jp == ["こんにちは。元気ですか？", "はい、元気です。"]


def test_chunk_processor_exceptions():
    processor = FixedLengthChunkProcessor(chunk_size=5)

    result = processor.process(None)
    assert result == []  # or whatever you expect for None  # Simulate invalid input
