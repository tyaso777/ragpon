import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from ragpon.domain.document_reader import (
    ExtensionBasedDocumentReaderFactory,
    HTMLReader,
    PDFReaderPyMuPDF,
    PDFReaderPyPDF,
    TXTReader,
)
from ragpon.domain.domain import ChunkSourceInfo

# or from ragpon.file_structure_definitions import PageInfo (if you need it)


@pytest.mark.parametrize(
    "file_path,reader_class",
    [
        ("test.pdf", PDFReaderPyMuPDF),
        ("test.txt", TXTReader),
        ("test.html", HTMLReader),
    ],
)
def test_extension_based_factory_valid(file_path, reader_class, mocker):
    """
    Test that ExtensionBasedDocumentReaderFactory returns correct reader classes
    for supported file extensions.
    """
    # Pretend the file path exists so the reader's constructor won't raise FileNotFoundError
    mocker.patch("os.path.exists", return_value=True)
    factory = ExtensionBasedDocumentReaderFactory()
    reader = factory.get_document_reader(file_path)
    assert isinstance(reader, reader_class)


def test_extension_based_factory_invalid_extension(mocker):
    """
    Test that ExtensionBasedDocumentReaderFactory raises ValueError for unsupported extensions.
    """
    mocker.patch("os.path.exists", return_value=True)
    factory = ExtensionBasedDocumentReaderFactory()
    with pytest.raises(ValueError, match="Unsupported file type"):
        factory.get_document_reader("unsupported.docx")


def test_pdf_reader_pypdf_invalid_path():
    """
    Test PDFReaderPyPDF raises FileNotFoundError for invalid file paths.
    """
    # Trying to create a PDFReaderPyPDF with a path that doesn't exist
    with pytest.raises(FileNotFoundError):
        PDFReaderPyPDF("nonexistent.pdf")


def test_pdf_reader_pymupdf_invalid_path():
    """
    Test PDFReaderPyMuPDF raises FileNotFoundError for invalid file paths.
    """
    with pytest.raises(FileNotFoundError):
        PDFReaderPyMuPDF("nonexistent.pdf")


def test_txt_reader_invalid_path():
    """
    Test TXTReader raises FileNotFoundError for invalid file paths.
    """
    with pytest.raises(FileNotFoundError):
        TXTReader("nonexistent.txt")


def test_html_reader_invalid_path():
    """
    Test HTMLReader raises FileNotFoundError for invalid file paths.
    """
    with pytest.raises(FileNotFoundError):
        HTMLReader("nonexistent.html")


def test_pdf_reader_pypdf_success(tmp_path, mocker):
    """
    Test that PDFReaderPyPDF reads pages successfully using pypdf.
    """
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.touch()  # Create an empty file so os.path.exists returns True

    # Mock pypdf's PdfReader to simulate two pages
    mock_pdf_reader = mocker.patch("pypdf.PdfReader", autospec=True)
    pdf_reader_instance = mock_pdf_reader.return_value
    page_mock_1 = MagicMock()
    page_mock_1.extract_text.return_value = "Page 1 content"
    page_mock_2 = MagicMock()
    page_mock_2.extract_text.return_value = "Page 2 content"
    pdf_reader_instance.pages = [page_mock_1, page_mock_2]

    reader = PDFReaderPyPDF(str(pdf_file))
    result = reader.read_document(start_page_number=10)

    assert len(result) == 2
    assert all(isinstance(item, ChunkSourceInfo) for item in result)
    assert result[0].text == "Page 1 content"
    assert result[0].metadata["page_number"] == 10
    assert result[1].text == "Page 2 content"
    assert result[1].metadata["page_number"] == 11

    mock_pdf_reader.assert_called_once_with(str(pdf_file))


def test_pdf_reader_pymupdf_success(tmp_path, mocker):
    """
    Test that PDFReaderPyMuPDF reads pages successfully using PyMuPDF (fitz).
    """
    pdf_file = tmp_path / "sample_mupdf.pdf"
    pdf_file.touch()

    mocked_fitz_open = mocker.patch("fitz.open", autospec=True)
    doc_mock = mocked_fitz_open.return_value
    page_mock_1 = MagicMock()
    page_mock_1.get_text.return_value = "MuPDF page 1"
    page_mock_2 = MagicMock()
    page_mock_2.get_text.return_value = "MuPDF page 2"
    doc_mock.__len__.return_value = 2
    doc_mock.__getitem__.side_effect = [page_mock_1, page_mock_2]

    reader = PDFReaderPyMuPDF(str(pdf_file))
    result = reader.read_document(start_page_number=5)

    assert len(result) == 2
    assert result[0].text == "MuPDF page 1"
    assert result[0].metadata["page_number"] == 5
    assert result[1].text == "MuPDF page 2"
    assert result[1].metadata["page_number"] == 6

    mocked_fitz_open.assert_called_once_with(str(pdf_file))
    page_mock_1.get_text.assert_called_once_with("text")
    page_mock_2.get_text.assert_called_once_with("text")


def test_txt_reader_success(tmp_path, mocker):
    """
    Test that TXTReader reads text from a file and uses chardet to detect encoding.
    """
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "Hello world", encoding="utf-8"
    )  # create a real file with some text

    mock_chardet = mocker.patch("chardet.detect", return_value={"encoding": "utf-8"})

    reader = TXTReader(str(txt_file))
    result = reader.read_document(start_page_number=2)

    assert len(result) == 1
    assert result[0].text == "Hello world"
    assert result[0].metadata["page_number"] == 2
    assert mock_chardet.called, "chardet.detect should have been called"


def test_html_reader_success(tmp_path, mocker):
    """
    Test that HTMLReader extracts text from HTML content.
    """
    html_file = tmp_path / "sample.html"
    html_content = "<html><body><h1>Title</h1><p>Some paragraph.</p></body></html>"
    html_file.write_text(html_content, encoding="utf-8")

    mock_chardet = mocker.patch("chardet.detect", return_value={"encoding": "utf-8"})

    reader = HTMLReader(str(html_file))
    result = reader.read_document()

    assert len(result) == 1
    # get_text(separator=" ", strip=True) likely yields "Title Some paragraph."
    assert "Title" in result[0].text
    assert "Some paragraph." in result[0].text
    assert result[0].metadata["page_number"] == 1
    mock_chardet.assert_called_once()


def test_logger_on_error(mocker):
    """
    Test that the logger captures error messages when file not found or IO error.
    """
    mock_logger = mocker.patch("ragpon.document_reader.logger")
    with pytest.raises(FileNotFoundError):
        _ = TXTReader("non_existent.txt")

    mock_logger.error.assert_called_with("File not found: non_existent.txt")
