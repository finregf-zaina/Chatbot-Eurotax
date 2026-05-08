import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class Document(BaseModel):
    text: str
    source: str = ""
    score: float = 0.0

class RerankerRequest(BaseModel):
    query: str
    documents: List[Document]
    top_n: int = 3

class RerankerResponse(BaseModel):
    reranked_documents: List[Document]
    total: int

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Reranker Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("RERANKER_USVC_PORT", 6004))

# ── Reranking simple par score de similarité ──────────────────────────────────
def simple_rerank(query: str, documents: List[Document], top_n: int) -> List[Document]:
    """
    Reranking simple basé sur la longueur et pertinence du texte.
    Sera remplacé par un vrai modèle de reranking quand disponible.
    """
    query_words = set(query.lower().split())
    
    scored = []
    for doc in documents:
        doc_words = set(doc.text.lower().split())
        overlap = len(query_words & doc_words)
        score = overlap / (len(query_words) + 1)
        scored.append(Document(
            text=doc.text,
            source=doc.source,
            score=round(score, 4)
        ))
    
    # Trier par score décroissant
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "reranker"}

@app.post("/v1/rerank", response_model=RerankerResponse)
async def rerank(request: RerankerRequest):
    try:
        reranked = simple_rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n
        )
        return RerankerResponse(
            reranked_documents=reranked,
            total=len(reranked)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)