from contextlib import nullcontext
from typing import Any

import ragpon.apps.streamlit.streamlit_app as streamlit_app
from ragpon.apps.streamlit.streamlit_app import (
    WARNING_LABELS,
    render_system_context_rows,
)


def test_empty_rows(monkeypatch: Any) -> None:
    captured: dict[str, str] = {}

    def fake_caption(msg: str) -> None:
        captured["caption"] = msg

    monkeypatch.setattr(streamlit_app.st, "caption", fake_caption)

    render_system_context_rows([], user_id="user1", session_id="session1")

    assert captured.get("caption") == WARNING_LABELS.NO_CONTEXT


def test_render_single_row_with_notes_link(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_markdown(md_str: str) -> None:
        captured["markdown"] = md_str

    def fake_text_area(**kwargs: Any) -> None:
        captured["text_area"] = kwargs

    monkeypatch.setattr(streamlit_app.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(streamlit_app.st, "markdown", fake_markdown)
    monkeypatch.setattr(streamlit_app.st, "text_area", fake_text_area)
    monkeypatch.setattr(streamlit_app.st, "divider", lambda: None)
    streamlit_app.st.session_state["selected_rag_rank_s_1"] = 1

    row = {
        "rag_rank": 1,
        "doc_id": "doc123",
        "semantic_distance": 0.123456,
        "notes_link": "notes://server/db/doc123",
        "text": "This is a test.",
    }

    render_system_context_rows([row], user_id="u", session_id="s", round_id=1)

    md_output = captured.get("markdown", "")
    assert "**Source:** doc123" in md_output
    assert "notes://server/db/doc123" in md_output
    assert captured["text_area"]["value"] == "This is a test."
