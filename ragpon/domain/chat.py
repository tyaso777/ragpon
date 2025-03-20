from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    """Allowed roles for a chat message."""

    user = "user"
    assistant = "assistant"


class SessionData(BaseModel):
    """
    Represents session information, including an ID, name, and privacy setting.
    """

    session_id: str = Field(..., description="The unique identifier for the session.")
    session_name: str = Field(..., description="The name of the session.")
    is_private_session: bool = Field(
        ..., description="Indicates if the session is private."
    )


class Message(BaseModel):
    """
    Represents a single chat message with role, content, message ID, and round ID.

    Attributes:
        role (RoleEnum): The role of the sender, either "user" or "assistant".
        content (str): The message content.
        id (str): A unique identifier for the message. (alias: id)
        round_id (int): The round number in the conversation.
        is_deleted (bool): Indicates if the message is deleted. Defaults to False.
    """

    role: RoleEnum = Field(
        ..., description="Role of the sender. Either 'user' or 'assistant'."
    )
    content: str = Field(..., description="The content of the message.")
    id: str = Field(..., alias="id", description="Unique identifier for the message.")
    round_id: int = Field(..., description="The round number of the conversation.")
    is_deleted: bool = Field(
        default=False,
        description="Indicates if the message has been marked as deleted.",
    )

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


class SessionCreate(BaseModel):
    """
    Represents the incoming data for creating a new session.

    Attributes:
        session_name (str): The name of the new session.
        is_private_session (bool): Whether the session is private.
        is_deleted (bool): Whether the session is marked as deleted.
    """

    session_name: str = Field(..., description="The new session's name/title.")
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

    session_name: str = Field(None, description="The new session name.")
    is_private_session: bool = Field(
        None, description="Whether the session is private."
    )
    is_deleted: bool = Field(
        None, description="Whether the session is marked as deleted."
    )


class DeleteRoundPayload(BaseModel):
    is_deleted: bool = Field(
        True, description="Whether the round is marked as deleted."
    )
    deleted_by: str = Field(
        ..., description="The user or assistant who marked the round as deleted."
    )


class PatchFeedbackPayload(BaseModel):
    feedback: Literal["good", "bad"] = Field(
        ..., description="The feedback message provided by the user."
    )
    reason: str | None = Field(None, description="The reason for the feedback, if any.")
