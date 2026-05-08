import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class SplitRequest(BaseModel):
    text: str
    filename: str = "document"
    chunk_size: int = 800
    chunk_overlap: int = 100

class Chunk(BaseModel):
    text: str
    source: str
    chunk_index: int

class SplitResponse(BaseModel):
    chunks: List[Chunk]
    total_chunks: int
    filename: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Text Splitter Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("TEXT_SPLITTER_USVC_PORT", 6001))

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "text_splitter"}

@app.post("/v1/split", response_model=SplitResponse)
async def split(request: SplitRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        texts = splitter.split_text(request.text)
        chunks = [
            Chunk(
                text=t,
                source=request.filename,
                chunk_index=i
            )
            for i, t in enumerate(texts)
        ]
        return SplitResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            filename=request.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)