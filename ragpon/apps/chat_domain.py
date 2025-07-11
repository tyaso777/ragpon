from datetime import datetime, timezone
from enum import Enum
from json import loads
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RoleEnum(str, Enum):
    """Allowed roles for a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RagModeEnum(str, Enum):
    """Enumerates Retrieval-Augmented Generation (RAG) operation modes.

    The enum values are used consistently across the Streamlit UI and
    FastAPI backend so that **one canonical string** identifies a mode
    everywhere (HTTP payloads, database, logs, etc.).

    Attributes
    ----------
    OPTIMIZED :
        _“RAG (Optimized Query)”_ – Performs an additional query-
        optimization step (e.g. rewriting or expansion) **before**
        retrieving documents.
    STANDARD :
        _“RAG (Standard)”_ – Retrieves documents with the original user
        query, without optimization.
    NO_RAG :
        _“No RAG”_ – Skips retrieval entirely; the LLM receives the raw
        user prompt only.
    """

    OPTIMIZED = "RAG (Optimized Query)"
    STANDARD = "RAG (Standard)"
    NO_RAG = "No RAG"

    @classmethod
    def list(cls) -> list[str]:
        """Return RAG-mode labels in declaration order.

        This thin wrapper exists mainly for UI widgets (e.g. a Streamlit
        ``st.radio``) that accept ``list[str]`` but not Enum members.

        Returns
        -------
        list[str]
            The human-readable enum *values*, preserving the order
            defined in the class.
        """
        return [m.value for m in cls]


class SessionData(BaseModel):
    """
    Represents session information, including an ID, name, and privacy setting.
    """

    session_id: str = Field(..., description="The unique identifier for the session.")
    session_name: str = Field(..., description="The name of the session.")
    is_private_session: bool = Field(
        ..., description="Indicates if the session is private."
    )
    last_touched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last user query time or session creation time.",
    )


class Message(BaseModel):
    """
    Represents a single chat message with role, content, message ID, and round ID.

    Attributes:
        role (RoleEnum): The role of the sender, either "user", "assistant", or "system".
        content (str): The message content. Must be JSON if role is "system".
        id (str): A unique identifier for the message. (alias: id)
        round_id (int): The round number in the conversation.
        is_deleted (bool): Indicates if the message is deleted. Defaults to False.
    """

    role: RoleEnum = Field(
        ..., description="Role of the sender. Either 'user', 'assistant', or 'system'."
    )
    content: str = Field(..., description="The content of the message.")
    id: str = Field(..., alias="id", description="Unique identifier for the message.")
    round_id: int = Field(..., description="The round number of the conversation.")
    is_deleted: bool = Field(
        default=False,
        description="Indicates if the message has been marked as deleted.",
    )

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_for_system_role(cls, v: str, info: ValidationInfo) -> str:
        """Ensure content is valid JSON if role is 'system'."""
        role = info.data.get("role") if info.data else None
        if role == RoleEnum.SYSTEM:
            try:
                loads(v)
            except Exception as exc:
                raise ValueError("System message content must be valid JSON") from exc
        return v

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "id": "6c72abf7-d494-4f7b-a383-40f5d3233726",
                "round_id": 0,
                "is_deleted": False,
            }
        }


class MessageListResponse(BaseModel):
    """
    Represents a paginated slice of chat history returned by FastAPI.

    Attributes:
        messages (list[Message]): Chronologically ordered chat messages
            (oldest → newest by ``round_id``).
        has_more (bool): Indicates that older rounds still exist on the server
            and can be fetched with a larger ``limit`` value.
    """

    messages: list[Message] = Field(
        ...,
        description="Chronological list of chat messages (oldest→newest).",
    )
    has_more: bool = Field(
        ...,
        description="True if additional (older) rounds can be requested.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!",
                        "id": "2c347f85-1bd1-46e7-950a-a7b05d4442a3",
                        "round_id": 42,
                        "is_deleted": False,
                    },
                    {
                        "role": "assistant",
                        "content": "Hi — how can I help you?",
                        "id": "6e82af89-77b4-4972-965b-3466d4fc5303",
                        "round_id": 42,
                        "is_deleted": False,
                    },
                ],
                "has_more": True,
            }
        }


class SessionCreate(BaseModel):
    """
    Represents the incoming data for creating a new session.

    Attributes:
        session_name (str): The name of the new session.
        is_private_session (bool): Whether the session is private.
        is_deleted (bool): Whether the session is marked as deleted.
    """

    session_name: str = Field(
        ...,
        max_length=30,
        description="The new session's name/title (up to 30 characters).",
    )
    is_private_session: bool = Field(
        False, description="Whether this session is private."
    )
    is_deleted: bool = Field(
        False, description="Whether this session is marked as deleted."
    )


class SessionUpdate(BaseModel):
    """
    Represents the incoming data to update a session.

    Attributes:
        session_name (str): The new session name.
        is_private_session (bool): Whether the session is private.
        is_deleted (bool): Whether the session is being marked as deleted.
    """

    session_name: str = Field(
        None, max_length=30, description="The new session name (up to 30 characters)."
    )
    is_private_session: bool = Field(
        None, description="Whether the session is private."
    )
    is_deleted: bool = Field(
        None, description="Whether the session is marked as deleted."
    )


class NewSessionData(BaseModel):
    """
    Represents a new session to be created, including session ID, name, and privacy status.

    Attributes:
        session_id (str): UUID generated client-side to uniquely identify the session.
        session_name (str): Name of the session.
        is_private_session (bool): Whether the session is private.
    """

    session_id: str = Field(..., description="UUID for the new session.")
    session_name: str = Field(
        ..., max_length=30, description="The name of the new session (up to 30 chars)."
    )
    is_private_session: bool = Field(..., description="Whether the session is private.")


class CreateSessionWithLimitRequest(BaseModel):
    """
    Request schema for creating a session with session count limit and consistency check.

    Attributes:
        new_session_data (NewSessionData): Information of the new session to create.
        known_session_ids (list[str]): Client-side known non-deleted session IDs.
        delete_target_session_id (Optional[str]): Session ID to delete if over limit.
    """

    new_session_data: NewSessionData = Field(..., description="New session details.")
    known_session_ids: list[str] = Field(
        ..., description="List of non-deleted session IDs the client is aware of."
    )
    delete_target_session_id: str | None = Field(
        None, description="Session ID to delete when session count is at max (10)."
    )


class SessionUpdateWithCheckRequest(BaseModel):
    """
    Request schema for updating a session with optimistic locking.

    This schema ensures that the session is only updated if the current database state
    matches the `before_` values provided by the client. Used to prevent overwriting changes
    made from other tabs or users.

    Attributes:
        before_session_name (str): The session name as known by the client before editing.
        before_is_private_session (bool): The session's privacy flag before editing.
        before_is_deleted (bool): Whether the session was marked as deleted before editing.
        after_session_name (str): The new name to update the session to.
        after_is_private_session (bool): The new privacy flag to update the session to.
        after_is_deleted (bool): Whether the session should be marked as deleted after editing.
    """

    before_session_name: str = Field(
        ..., max_length=30, description="Session name before editing."
    )
    before_is_private_session: bool = Field(
        ..., description="Privacy status before editing."
    )
    before_is_deleted: bool = Field(..., description="Deletion status before editing.")
    after_session_name: str = Field(
        ..., max_length=30, description="New session name to apply."
    )
    after_is_private_session: bool = Field(
        ..., description="New privacy status to apply."
    )
    after_is_deleted: bool = Field(..., description="New deletion status to apply.")


class DeleteRoundPayload(BaseModel):
    is_deleted: bool = Field(
        True, description="Whether the round is marked as deleted."
    )
    deleted_by: str = Field(
        ..., description="The user or assistant who marked the round as deleted."
    )


class PatchFeedbackPayload(BaseModel):
    user_id: str = Field(..., description="The user who submitted the feedback.")
    session_id: str = Field(
        ..., description="The session in which the message was generated."
    )
    feedback: Literal["good", "bad"] = Field(
        ..., description="The feedback type provided by the user."
    )
    reason: str | None = Field(None, description="The reason for the feedback, if any.")
