import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    total: int
    model: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Embeddings Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("EMBEDDING_USVC_PORT", 6012))
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# ── Chargement du modèle au démarrage ─────────────────────────────────────────
print(f"⏳ Chargement du modèle : {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print(f"✅ Modèle chargé !")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "embeddings",
        "model": MODEL_NAME
    }

@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Liste de textes vide.")
    try:
        vectors = embedding_model.encode(
            request.texts,
            show_progress_bar=False
        ).tolist()

        return EmbedResponse(
            embeddings=vectors,
            total=len(vectors),
            model=MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)