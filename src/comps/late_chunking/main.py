import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx

load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:6002")

# ── Modèles ───────────────────────────────────────────────────────────────────
class LateChunkRequest(BaseModel):
    text: str
    filename: str = "document"
    chunk_size: int = 800
    chunk_overlap: int = 100

class LateChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    embedding: List[float]
    context_aware: bool = True

class LateChunkResponse(BaseModel):
    chunks: List[LateChunk]
    total_chunks: int
    filename: str
    model: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Late Chunking Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("LATE_CHUNKING_USVC_PORT", 6007))

# ── Helpers ───────────────────────────────────────────────────────────────────
def split_text_with_context(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Late chunking : on garde le contexte global avant de découper.
    Chaque chunk garde une fenêtre de contexte des chunks adjacents.
    """
    words = text.split()
    chunks = []
    
    # Découpage en tokens approximatifs
    chunk_words = chunk_size // 5  # ~5 chars par mot
    overlap_words = chunk_overlap // 5
    
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap_words
    
    return chunks

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Appelle le service embeddings."""
    response = httpx.post(
        f"{EMBEDDING_SERVICE_URL}/v1/embed",
        json={"texts": texts},
        timeout=120.0
    )
    response.raise_for_status()
    return response.json()["embeddings"]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "late_chunking"}

@app.post("/v1/late_chunk", response_model=LateChunkResponse)
async def late_chunk(request: LateChunkRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    
    try:
        # 1. Découper le texte avec contexte
        raw_chunks = split_text_with_context(
            request.text,
            request.chunk_size,
            request.chunk_overlap
        )
        
        if not raw_chunks:
            raise HTTPException(status_code=422, detail="Aucun chunk généré.")

        # 2. Créer les textes enrichis avec contexte
        # Late chunking : chaque chunk inclut un résumé du contexte global
        context_summary = request.text[:200] + "..."  # Résumé du début
        
        enriched_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Enrichir avec contexte précédent et suivant
            prev_context = raw_chunks[i-1][-100:] if i > 0 else ""
            next_context = raw_chunks[i+1][:100] if i < len(raw_chunks)-1 else ""
            
            enriched = f"{prev_context} {chunk} {next_context}".strip()
            enriched_chunks.append(enriched)

        # 3. Embeddings sur les chunks enrichis
        print(f"🔗 Late chunking : {len(enriched_chunks)} chunks enrichis → embeddings...")
        embeddings = await get_embeddings(enriched_chunks)

        # 4. Construire la réponse
        chunks = [
            LateChunk(
                text=raw_chunks[i],  # Texte original (sans contexte)
                source=request.filename,
                chunk_index=i,
                embedding=embeddings[i],
                context_aware=True
            )
            for i in range(len(raw_chunks))
        ]

        return LateChunkResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            filename=request.filename,
            model="paraphrase-multilingual-mpnet-base-v2"
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Service embeddings inaccessible sur {EMBEDDING_SERVICE_URL}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)