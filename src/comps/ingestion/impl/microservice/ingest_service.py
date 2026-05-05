import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()


def run_ingestion(force_recreate: bool = False):
    """
    Charge les documents du dossier /data, les découpe, les vectorise
    et les indexe dans Qdrant.

    Args:
        force_recreate: Si True, recrée la collection (écrase l'existant).
                        Mettre False en prod pour ingestion incrémentale.
    """
    directory_path = "./data"
    print(f"\n📁 Recherche de documents dans '{directory_path}'...")

    # ✅ Validation des variables d'environnement critiques
    required_vars = [
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "QDRANT_URL"
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"❌ Variables .env manquantes : {', '.join(missing)}\n"
            f"   Vérifie ton fichier .env à la racine du projet."
        )

    # ---- 1. Chargement des documents ----------------------------------------
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

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
                # ✅ Ajout du nom de fichier dans les métadonnées pour traçabilité
                for doc in loaded:
                    doc.metadata["source"] = file
                docs.extend(loaded)
                print(f"     ✅ {len(loaded)} page(s) chargée(s)")
            except Exception as e:
                print(f"     ⚠️  Erreur sur {file} : {e}")

    if not docs:
        print("❌ Aucun document PDF ou DOCX trouvé dans /data. Arrêt.")
        return

    print(f"\n📊 Total : {len(docs)} page(s) chargée(s) depuis {directory_path}")

    # ---- 2. Découpage (Chunking) ---------------------------------------------
    # chunk_size=800 / overlap=100 : bon équilibre pour docs juridiques/fiscaux
    # ✅ Séparateurs adaptés aux documents français avec numérotation légale
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✂️  Texte découpé en {len(splits)} chunks.")

    # ---- 3. Embeddings Azure AI Search --------------------------------------
    print("\n🔗 Initialisation des Embeddings Azure AI Search...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # ---- 4. Ingestion dans Qdrant -------------------------------------------
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "eurotax_docs")

    print(f"📡 Connexion à Qdrant : {qdrant_url}")
    print(f"   Collection cible  : '{collection_name}'")
    print(f"   Mode              : {'🔄 Recréation complète' if force_recreate else '➕ Ajout incrémental'}")

    try:
        vectorstore = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),  # None si Qdrant local
            collection_name=collection_name,
            force_recreate=force_recreate
        )
        print(f"\n✅ SUCCÈS : {len(splits)} chunks indexés dans Qdrant !")
        print(f"   Collection '{collection_name}' prête pour le chatbot.\n")

    except Exception as e:
        print(f"\n❌ Erreur Qdrant : {e}")
        raise


if __name__ == "__main__":
    # ✅ En dev : force_recreate=True pour repartir propre à chaque test
    # ✅ En prod : force_recreate=False pour ingestion incrémentale
    run_ingestion(force_recreate=True)
