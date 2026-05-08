@echo off
echo Demarrage des services Python RAG Eurotax...

start "Text Extractor :6010" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.text_extractor.main:app --host 0.0.0.0 --port 6010 --reload"
timeout /t 3

start "Text Splitter :6011" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.text_splitter.main:app --host 0.0.0.0 --port 6011 --reload"
timeout /t 3

start "Embeddings :6002" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.embeddings.main:app --host 0.0.0.0 --port 6002 --reload"
timeout /t 5

start "Retriever :6003" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.retrievers.impl.microservice.retriever_service:app --host 0.0.0.0 --port 6003 --reload"
timeout /t 3

start "Reranker :6004" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.reranks.impl.microservice.rerank_service:app --host 0.0.0.0 --port 6004 --reload"
timeout /t 3

start "LLM :6005" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.llms.impl.microservice.llm_service:app --host 0.0.0.0 --port 6005 --reload"
timeout /t 3

start "API Principal :8000" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

start "Guardrails :6006" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.guardrails.main:app --host 0.0.0.0 --port 6006 --reload"
timeout /t 3

start "Late Chunking :6007" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.late_chunking.main:app --host 0.0.0.0 --port 6007 --reload"
timeout /t 3

start "DocSum :6008" cmd /k "cd /d D:\projets\Git\chatbot-eurotax && venv\Scripts\activate && uvicorn src.comps.docsum.main:app --host 0.0.0.0 --port 6008 --reload"
timeout /t 3

echo Tous les services demarres !
pause