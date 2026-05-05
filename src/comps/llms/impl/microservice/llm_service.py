import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.comps.retrievers.impl.microservice.retriever_service import get_retriever

load_dotenv()

# ✅ Prompt correctement structuré pour RetrievalQA (chain_type="stuff")
# {context} = les chunks récupérés par Qdrant
# {question} = la question de l'utilisateur
PROMPT_TEMPLATE = """
Tu es l'assistant IA officiel d'Eurotax. Ton rôle est d'orienter les collaborateurs
sur les processus internes (RH, IT, Administratif, Accises).

Règles strictes :
1. Utilise UNIQUEMENT les informations contenues dans le contexte fourni ci-dessous.
2. Si la réponse n'est pas dans le contexte, réponds poliment que tu ne disposes pas
   de cette information et suggère de contacter le point de contact (POC) concerné.
3. Reste professionnel, précis et bienveillant.
4. Réponds toujours en français.

Contexte extrait des documents Eurotax :
-----------------------------------------
{context}
-----------------------------------------

Question du collaborateur : {question}

Réponse :"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def get_eurotax_response(query: str) -> str:
    """
    Prend une question en entrée et retourne la réponse RAG du chatbot Eurotax.
    """
    # 1. LLM Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

    # 2. Retriever Qdrant
    retriever = get_retriever()

    # 3. Chaîne RAG avec prompt personnalisé
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # ✅ Utile pour debug / traçabilité
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # 4. Appel et extraction de la réponse
    result = qa_chain.invoke({"query": query})  # ✅ .invoke() remplace .run() (déprécié)

    answer = result.get("result", "Je n'ai pas pu générer de réponse.")
    sources = result.get("source_documents", [])

    # 5. Affichage des sources (optionnel, utile en dev)
    if sources:
        print("\n📄 Sources utilisées :")
        for doc in sources:
            src = doc.metadata.get("source", "Inconnu")
            page = doc.metadata.get("page", "?")
            print(f"  - {src} (page {page})")

    return answer


if __name__ == "__main__":
    question = "Quels sont les taux d'accise pour l'alcool en France en 2026 ?"
    print(f"\n❓ Question : {question}\n")
    try:
        reponse = get_eurotax_response(question)
        print(f"\n✅ Réponse Eurotax :\n{reponse}")
    except Exception as e:
        print(f"❌ Erreur : {e}")