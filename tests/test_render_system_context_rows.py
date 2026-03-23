from contextlib import nullcontext
from typing import Any

import ragpon.apps.streamlit.streamlit_app as streamlit_app
from ragpon.apps.streamlit.streamlit_app import (
    LABELS,
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


def test_filtered_rows_show_no_context(monkeypatch: Any) -> None:
    captured: dict[str, str] = {}

    def fake_caption(msg: str) -> None:
        captured["caption"] = msg

    monkeypatch.setattr(streamlit_app.st, "caption", fake_caption)
    streamlit_app.st.session_state["selected_rag_rank_s_2"] = 99

    row = {
        "rag_rank": 1,
        "doc_id": "doc123",
        "text": "This is a test.",
    }

    render_system_context_rows([row], user_id="u", session_id="s", round_id=2)

    assert captured.get("caption") == WARNING_LABELS.NO_CONTEXT


def test_assistant_without_context_rows_shows_only_no_context(monkeypatch: Any) -> None:
    captured: dict[str, list[str]] = {"captions": [], "markdowns": []}

    class DummyChatContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    def fake_chat_message(*args: Any, **kwargs: Any) -> DummyChatContext:
        return DummyChatContext()

    def fake_columns(spec: Any) -> list[Any]:
        class DummyColumn:
            def button(self, *args: Any, **kwargs: Any) -> bool:
                return False

        return [DummyColumn() for _ in range(len(spec))]

    monkeypatch.setattr(streamlit_app.st, "chat_message", fake_chat_message)
    monkeypatch.setattr(streamlit_app.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "columns", fake_columns)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda msg: captured["captions"].append(msg))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda msg, **kwargs: captured["markdowns"].append(msg))
    monkeypatch.setattr(streamlit_app.st, "container", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(streamlit_app.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(streamlit_app.st, "warning", lambda *args, **kwargs: None)

    messages = [
        streamlit_app.Message(
            role=streamlit_app.RoleEnum.ASSISTANT,
            content="回答本文です",
            id="m1",
            round_id=1,
            is_deleted=False,
        )
    ]

    render_chat_messages = streamlit_app.render_chat_messages
    render_chat_messages(
        messages=messages,
        server_url="http://localhost:8000",
        user_id="u",
        current_session_id="s",
        disabled_ui=False,
    )

    assert WARNING_LABELS.NO_CONTEXT in captured["captions"]
    assert LABELS.VIEW_SOURCES not in captured["markdowns"]
