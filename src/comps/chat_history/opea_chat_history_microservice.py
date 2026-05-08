import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pyodbc
import uuid
from datetime import datetime

load_dotenv()

# ── Config SQL Server ─────────────────────────────────────────────────────────
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DATABASE = os.getenv("SQL_DATABASE", "ChatHistory")
SQL_USERNAME = os.getenv("SQL_USERNAME", "sa")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")
USVC_PORT = int(os.getenv("CHAT_HISTORY_USVC_PORT", 6020))

def get_connection():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD};"
    )
    return pyodbc.connect(conn_str)

# ── Modèles ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    question: str
    answer: str

class ChatHistory(BaseModel):
    id: Optional[str] = None
    history: List[ChatMessage]

class ChatHistoryName(BaseModel):
    id: str
    history_name: str

# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Créer la table si elle n'existe pas
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ChatHistories' AND xtype='U')
            CREATE TABLE ChatHistories (
                id NVARCHAR(50) PRIMARY KEY,
                user_id NVARCHAR(255) NOT NULL,
                history_name NVARCHAR(250) NOT NULL,
                history NVARCHAR(MAX) NOT NULL,
                created_at DATETIME DEFAULT GETDATE()
            )
        """)
        conn.commit()
        conn.close()
        print("✅ SQL Server connecté et table prête")
    except Exception as e:
        print(f"⚠️ SQL Server non disponible : {e}")
    yield

app = FastAPI(
    title="Chat History Service",
    description="Service historique avec SQL Server",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/v1/health_check")
def health_check():
    return {"status": "ok", "service": "chat_history", "db": "SQL Server"}

@app.post("/v1/chat_history/save", response_model=ChatHistoryName)
async def save_history(document: ChatHistory, request: Request):
    import json
    try:
        conn = get_connection()
        cursor = conn.cursor()
        history_json = json.dumps([m.dict() for m in document.history])
        history_name = document.history[0].question[:30]

        if document.id:
            cursor.execute(
                "UPDATE ChatHistories SET history=? WHERE id=?",
                history_json, document.id
            )
            conn.commit()
            return ChatHistoryName(id=document.id, history_name=history_name)
        else:
            new_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO ChatHistories (id, user_id, history_name, history) VALUES (?,?,?,?)",
                new_id, "default_user", history_name, history_json
            )
            conn.commit()
            return ChatHistoryName(id=new_id, history_name=history_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/v1/chat_history/get")
async def get_history(history_id: Optional[str] = None):
    import json
    try:
        conn = get_connection()
        cursor = conn.cursor()
        if history_id:
            cursor.execute("SELECT * FROM ChatHistories WHERE id=?", history_id)
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="History not found")
            return {"id": row[0], "history": json.loads(row[3]), "history_name": row[2]}
        else:
            cursor.execute("SELECT id, history_name FROM ChatHistories")
            rows = cursor.fetchall()
            return [{"id": r[0], "history_name": r[1]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/v1/chat_history/delete")
async def delete_history(history_id: str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ChatHistories WHERE id=?", history_id)
        conn.commit()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)