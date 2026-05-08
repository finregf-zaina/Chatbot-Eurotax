
import os

# Qdrant config - sera rempli quand la clé arrive
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")

# TODO: activer quand la clé Qdrant est disponible
# from qdrant_client import QdrantClient
# client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
