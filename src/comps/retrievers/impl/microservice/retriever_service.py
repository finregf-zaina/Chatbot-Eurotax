import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient

load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:6002")

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

class RetrieverRequest(BaseModel):
    query: str
    k: int = 4

class DocumentResult(BaseModel):
    text: str
    score: float
    source: str

class RetrieverResponse(BaseModel):
    documents: List[DocumentResult]
    total: int

app = FastAPI(title="Retriever Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

USVC_PORT = int(os.getenv("RETRIEVER_USVC_PORT", 6003))

def get_retriever(k: int = 4):
    embeddings = RemoteEmbeddings()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "eurotax_docs"),
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

@app.get("/health")
async def health():
    return {"status": "ok", "service": "retriever"}

@app.post("/v1/retrieve", response_model=RetrieverResponse)
async def retrieve(request: RetrieverRequest):
    try:
        retriever = get_retriever(k=request.k)
        docs = retriever.invoke(request.query)
        documents = [
            DocumentResult(
                text=doc.page_content,
                score=1.0,
                source=doc.metadata.get("source", "Inconnu")
            )
            for doc in docs
        ]
        return RetrieverResponse(documents=documents, total=len(documents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)