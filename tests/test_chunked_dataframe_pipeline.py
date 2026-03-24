import pandas as pd

from ragpon.chunk_processor import FixedLengthChunkProcessor
from ragpon.domain.document_processing_pipeline import (
    ChunkedDataFrameDocumentProcessingPipeline,
)
from ragpon.domain.domain import BaseDocument
from ragpon.domain.metadata_generator import CustomMetadataGenerator


def test_chunked_dataframe_pipeline_splits_rows_and_preserves_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                "source_doc_id": "doc-1",
                "database_title": "社内規程DB",
                "body_text": "abcdefghij",
                "file_path": "/tmp/sample.json",
                "page_number": 1,
                "notes_link": "notes://sample",
                "category_1": "規程",
            }
        ]
    )

    pipeline = ChunkedDataFrameDocumentProcessingPipeline(
        chunk_processor=FixedLengthChunkProcessor(chunk_size=4),
        metadata_generator=CustomMetadataGenerator(BaseDocument),
        chunk_col_name="body_text",
        id_col_name="source_doc_id",
    )

    chunks, metadata = pipeline.process_document(df=df)

    assert chunks == ["abcd", "efgh", "ij"]
    assert [m.doc_id for m in metadata] == ["doc-1_No.0", "doc-1_No.1", "doc-1_No.2"]
    assert [m.metadata["serial_number"] for m in metadata] == [0, 1, 2]
    assert [m.metadata["chunk_index_in_page"] for m in metadata] == [0, 1, 2]
    assert all(m.metadata["page_number"] == 1 for m in metadata)
    assert all(m.metadata["file_path"] == "/tmp/sample.json" for m in metadata)
    assert all(m.metadata["notes_link"] == "notes://sample" for m in metadata)
    assert all(m.metadata["source_doc_id"] == "doc-1" for m in metadata)
    assert all(m.metadata["database_title"] == "社内規程DB" for m in metadata)
    assert all(m.metadata["category_1"] == "規程" for m in metadata)
    assert all("body_text" not in m.metadata for m in metadata)
