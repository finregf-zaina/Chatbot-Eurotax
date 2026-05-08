from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from .documents import ChatHistoryDocument

class OPEAChatHistoryConnector:
    def __init__(self, mongodb_host: str, mongodb_port: str) -> None:
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.client = None

    async def init_async(self):
        self.client = AsyncIOMotorClient(f"mongodb://{self.mongodb_host}:{self.mongodb_port}")
        await init_beanie(database=self.client["CHAT_HISTORY"], document_models=[ChatHistoryDocument])

    async def close(self):
        if self.client:
            self.client.close()

    def _generate_title(self, history) -> str:
        q = history[0].question
        return q[:30] if len(q) > 30 else q

    async def create_new_history(self, history, user_id: str):
        h = ChatHistoryDocument(
            history=history,
            user_id=user_id,
            history_name=self._generate_title(history)
        )
        await h.insert()
        return {"id": str(h.id), "history_name": h.history_name}

    async def append_history(self, history_id: str, history, user_id: str):
        h = await ChatHistoryDocument.get(history_id)
        if not h:
            raise ValueError(f"History {history_id} not found.")
        if h.user_id != user_id:
            raise ValueError("User ID mismatch.")
        h.history.extend(history)
        await h.save()
        return {"id": str(h.id), "history_name": h.history_name}

    async def change_history_name(self, history_id: str, history_name: str, user_id: str):
        if not history_name or history_name.strip() == "":
            raise ValueError("History name cannot be empty.")
        if len(history_name) > 250:
            raise ValueError("History name too long.")
        h = await ChatHistoryDocument.get(history_id)
        if not h:
            raise ValueError(f"History {history_id} not found.")
        if h.user_id != user_id:
            raise ValueError("User ID mismatch.")
        h.history_name = history_name
        await h.save()

    async def delete_history(self, history_id: str, user_id: str):
        h = await ChatHistoryDocument.get(history_id)
        if not h:
            raise ValueError(f"History {history_id} not found.")
        if h.user_id != user_id:
            raise ValueError("User ID mismatch.")
        await h.delete()

    async def get_all_histories_for_user(self, user_id: str):
        histories = await ChatHistoryDocument.find(
            ChatHistoryDocument.user_id == user_id
        ).to_list()
        return [{"id": str(h.id), "history_name": h.history_name} for h in histories]

    async def get_history_by_id(self, history_id: str, user_id: str):
        h = await ChatHistoryDocument.get(history_id)
        if not h:
            raise ValueError(f"History {history_id} not found.")
        if h.user_id != user_id:
            raise ValueError("User ID mismatch.")
        return h
