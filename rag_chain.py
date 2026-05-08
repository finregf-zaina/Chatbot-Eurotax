"""
rag_chain.py — Orchestrateur principal du chatbot RAG Eurotax.

Usage direct :
    python rag_chain.py

Ou importé depuis api.py :
    from rag_chain import ask_eurotax
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:6005")

def ask_eurotax(question: str) -> str:
    if not question or not question.strip():
        return "Veuillez poser une question valide."
    try:
        response = httpx.post(
            f"{LLM_SERVICE_URL}/v1/chat",
            json={"question": question.strip()},
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["answer"]
    except httpx.TimeoutException:
        return "Le service met trop de temps à répondre. Réessayez."
    except Exception as e:
        return f"Erreur : {str(e)}"

if __name__ == "__main__":
    print("=" * 60)
    print("   Chatbot Eurotax — Mode test")
    print("=" * 60)
    while True:
        question = input("\nVotre question : ").strip()
        if question.lower() in ["quitter", "exit", "quit"]:
            print("Au revoir !")
            break
        if not question:
            continue
        print("\nRecherche en cours...")
        reponse = ask_eurotax(question)
        print(f"\nRéponse :\n{reponse}")