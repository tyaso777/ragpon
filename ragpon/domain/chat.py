from enum import Enum

from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    """Allowed roles for a chat message."""

    user = "user"
    assistant = "assistant"


class SessionData(BaseModel):
    """
    Represents session information, including an ID, name, and privacy setting.
    """

    session_id: str
    session_name: str
    is_private_session: bool


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
