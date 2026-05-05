import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
load_dotenv()


def get_retriever(k: int = 4):
    """
    Initialise et retourne un Retriever connecté à Qdrant + Azure Embeddings.
    
    Args:
        k: Nombre de chunks à récupérer (défaut=4, plus safe que 3)
    
    Returns:
        Un retriever LangChain prêt à l'emploi.
    """
    # 1. Embeddings Azure AI Search
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # 2. Client Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "eurotax_docs")

    # ✅ Validation des variables critiques
    if not qdrant_url:
        raise ValueError("❌ QDRANT_URL manquante dans le fichier .env")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,  # None si Qdrant local sans auth
        timeout=30
    )

    # ✅ Vérification que la collection existe avant de continuer
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            raise RuntimeError(
                f"❌ La collection '{collection_name}' n'existe pas dans Qdrant.\n"
                f"   Lance d'abord : python ingest.py"
            )
    except UnexpectedResponse as e:
        raise ConnectionError(f"❌ Impossible de joindre Qdrant : {e}")

    # 3. Vector Store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # 4. Retriever avec recherche par similarité
    # search_type="similarity" par défaut
    # ✅ On peut passer à "mmr" (Max Marginal Relevance) pour plus de diversité
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever


if __name__ == "__main__":
    print("🔍 Test du Retriever Eurotax...")
    try:
        r = get_retriever()
        print("✅ SUCCÈS : Retriever connecté à Azure Embeddings + Qdrant !")

        # Test de recherche rapide
        test_query = "taux accise alcool France"
        docs = r.invoke(test_query)
        print(f"\n📄 {len(docs)} chunks trouvés pour : '{test_query}'")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:200], "...")
    except Exception as e:
        print(f"❌ Erreur : {e}")