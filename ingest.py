"""
ingest.py — Point d'entrée pour l'ingestion des documents Eurotax.

Usage :
    python ingest.py              → ingestion incrémentale (ajout)
    python ingest.py --recreate   → recrée la collection Qdrant (tout écraser)
"""

import argparse
from src.comps.ingestion.impl.microservice.ingest_service import run_ingestion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion des documents Eurotax vers Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recrée la collection Qdrant from scratch (efface l'existant)"
    )
    args = parser.parse_args()

    run_ingestion(force_recreate=args.recreate)