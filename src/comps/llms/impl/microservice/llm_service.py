import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient

load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:6002")

# ── RemoteEmbeddings ──────────────────────────────────────────────────────────
class RemoteEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = httpx.post(
            f"{EMBEDDING_SERVICE_URL}/v1/embed",
            json={"texts": texts},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        response = httpx.post(
            f"{EMBEDDING_SERVICE_URL}/v1/embed",
            json={"texts": [text]},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

# ── Modèles ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history_id: Optional[str] = None

class SourceDocument(BaseModel):
    source: str
    page: int = 0

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = []
    history_id: Optional[str] = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

USVC_PORT = int(os.getenv("LLM_USVC_PORT", 6005))

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
Tu es l'assistant IA officiel d'Eurotax. Ton rôle est d'orienter les collaborateurs
sur les processus internes (RH, IT, Administratif, Accises).

Règles strictes :
1. Utilise UNIQUEMENT les informations contenues dans le contexte fourni.
2. Si la réponse n'est pas dans le contexte, réponds poliment que tu ne disposes
   pas de cette information et suggère de contacter le POC concerné.
3. Reste professionnel, précis et bienveillant.
4. Réponds toujours en français.

Contexte :
-----------------------------------------
{context}
-----------------------------------------

Question : {question}

Réponse :"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_retriever(k: int = 4):
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "eurotax_docs"),
        embedding=RemoteEmbeddings()
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def get_eurotax_response(question: str) -> dict:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    result = qa_chain.invoke({"query": question})
    answer = result.get("result", "Je n'ai pas pu générer de réponse.")
    sources = [
        SourceDocument(
            source=doc.metadata.get("source", "Inconnu"),
            page=doc.metadata.get("page", 0)
        )
        for doc in result.get("source_documents", [])
    ]
    return {"answer": answer, "sources": sources}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "llm"}

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")
    try:
        result = get_eurotax_response(request.question.strip())
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            history_id=request.history_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)