from beanie import Document
from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    question: str
    answer: str

class ChatHistoryDocument(Document):
    history: List[ChatMessage]
    user_id: str
    history_name: str

    class Settings:
        name = "chat_history_collection"