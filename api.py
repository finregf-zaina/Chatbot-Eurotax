"""
api.py — Point d'entrée FastAPI du chatbot Eurotax.

Lancement :
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints :
    GET  /          → health check
    POST /chat      → poser une question au chatbot
    GET  /docs      → documentation Swagger automatique
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import ask_eurotax
import os

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chatbot Eurotax",
    description="Assistant IA interne Eurotax — RAG sur documents fiscaux/RH/IT",
    version="1.0.0"
)

# ── CORS (pour que le frontend de ta collègue puisse appeler l'API) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ✅ En prod, remplacer par l'URL du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modèles de données ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quels sont les taux d'accise pour l'alcool en France en 2026 ?"
            }
        }

class ChatResponse(BaseModel):
    question: str
    answer: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return HealthResponse(
        status="ok",
        service="Chatbot Eurotax",
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Envoie une question au chatbot Eurotax.
    
    - **question** : La question posée par le collaborateur (en français)
    
    Retourne la réponse générée par le pipeline RAG (Azure OpenAI + Qdrant).
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide."
        )

    try:
        answer = ask_eurotax(request.question)
        return ChatResponse(
            question=request.question,
            answer=answer
        )
    except Exception as e:
        # ✅ Log l'erreur côté serveur sans exposer les détails en prod
        print(f"[ERREUR /chat] {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse : {str(e)}"
        )

# ── Lancement direct ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True   # ✅ Désactiver en prod
    )