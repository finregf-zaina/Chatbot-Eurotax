import os
import httpx
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings

load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:6002")

# ── Embeddings via microservice ───────────────────────────────────────────────
class RemoteEmbeddings(Embeddings):
    """Appelle le microservice embeddings par lots pour éviter les timeouts."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 10  # ✅ Par lots de 10
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"   📦 Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} textes)...")
            response = httpx.post(
                f"{EMBEDDING_SERVICE_URL}/v1/embed",
                json={"texts": batch},
                timeout=120.0  # ✅ 2 minutes par batch
            )
            response.raise_for_status()
            all_embeddings.extend(response.json()["embeddings"])
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        response = httpx.post(
            f"{EMBEDDING_SERVICE_URL}/v1/embed",
            json={"texts": [text]},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def run_ingestion(force_recreate: bool = False):
    directory_path = "./data"
    print(f"\n📁 Recherche de documents dans '{directory_path}'...")

    # ✅ Validation
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"❌ Variables .env manquantes : {', '.join(missing)}")

    # ---- 1. Chargement -------------------------------------------------------
    loaders = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}
    docs = []
    os.makedirs(directory_path, exist_ok=True)

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            print(f"  📄 Chargement : {file}")
            try:
                loader = loaders[ext](file_path)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = file
                docs.extend(loaded)
                print(f"     ✅ {len(loaded)} page(s)")
            except Exception as e:
                print(f"     ⚠️ Erreur : {e}")

    if not docs:
        print("❌ Aucun document trouvé.")
        return

    print(f"\n📊 Total : {len(docs)} page(s)")

    # ---- 2. Découpage --------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = splitter.split_documents(docs)
    print(f"✂️  {len(splits)} chunks.")

    # ---- 3. Embeddings via microservice --------------------------------------
    print(f"\n🔗 Connexion au service embeddings : {EMBEDDING_SERVICE_URL}")
    try:
        health = httpx.get(f"{EMBEDDING_SERVICE_URL}/health", timeout=10)
        health.raise_for_status()
        print("✅ Service embeddings disponible !")
    except Exception:
        raise ConnectionError(
            f"❌ Service embeddings inaccessible sur {EMBEDDING_SERVICE_URL}\n"
            f"   Lance d'abord : uvicorn src.comps.embeddings.main:app --port 6012"
        )

    embeddings = RemoteEmbeddings()

    # ---- 4. Ingestion Qdrant -------------------------------------------------
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "eurotax_docs")

    print(f"\n📡 Connexion Qdrant : {qdrant_url}")
    print(f"   Collection : '{collection_name}'")
    print(f"   Mode : {'🔄 Recréation' if force_recreate else '➕ Incrémental'}")

    try:
        QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            force_recreate=force_recreate
        )
        print(f"\n✅ SUCCÈS : {len(splits)} chunks indexés !")
        print(f"   Collection '{collection_name}' prête.\n")
    except Exception as e:
        print(f"\n❌ Erreur Qdrant : {e}")
        raise


if __name__ == "__main__":
    run_ingestion(force_recreate=True)
    
