"""
rag_chain.py — Orchestrateur principal du chatbot RAG Eurotax.

Usage direct :
    python rag_chain.py

Ou importé depuis api.py :
    from rag_chain import ask_eurotax
"""

from src.comps.llms.impl.microservice.llm_service import get_eurotax_response


def ask_eurotax(question: str) -> str:
    """
    Fonction principale appelée par l'API ou en direct.
    Prend une question et retourne la réponse du chatbot Eurotax.

    Args:
        question: La question posée par le collaborateur.

    Returns:
        La réponse générée par le pipeline RAG.
    """
    if not question or not question.strip():
        return "❌ Veuillez poser une question valide."

    return get_eurotax_response(question.strip())


if __name__ == "__main__":
    print("=" * 60)
    print("   🤖 Chatbot Eurotax — Mode test interactif")
    print("   Tape 'quitter' pour arrêter")
    print("=" * 60)

    while True:
        question = input("\n❓ Votre question : ").strip()

        if question.lower() in ["quitter", "exit", "quit"]:
            print("👋 Au revoir !")
            break

        if not question:
            continue

        print("\n⏳ Recherche en cours...")
        try:
            reponse = ask_eurotax(question)
            print(f"\n✅ Réponse :\n{reponse}")
        except Exception as e:
            print(f"\n❌ Erreur : {e}")