import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
import docx2txt

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class SummarizeTextRequest(BaseModel):
    text: str
    language: str = "fr"
    max_length: int = 500
    style: str = "professional"  # professional, bullet_points, brief

class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    filename: Optional[str] = None
    language: str
    style: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Document Summarization Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("DOCSUM_USVC_PORT", 6008))

# ── Prompts par style ─────────────────────────────────────────────────────────
PROMPTS = {
    "professional": """
Tu es un expert en synthèse de documents professionnels.
Résume le document suivant en français de manière professionnelle et concise.
Maximum {max_length} mots.

Document :
{text}

Résumé professionnel :""",

    "bullet_points": """
Tu es un expert en synthèse de documents.
Résume le document suivant en français sous forme de points clés (bullet points).
Maximum {max_length} mots au total.

Document :
{text}

Points clés :
- """,

    "brief": """
Résume ce document en français en 2-3 phrases maximum.

Document :
{text}

Résumé bref :"""
}

# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.3
    )

def extract_text_from_file(file_path: str, ext: str) -> str:
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        return docx2txt.process(file_path)
    return ""

async def summarize_text(text: str, style: str, max_length: int) -> str:
    if style not in PROMPTS:
        style = "professional"
    
    prompt_template = PROMPTS[style]
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text", "max_length"]
    )
    
    llm = get_llm()
    chain = prompt | llm
    
    result = await chain.ainvoke({
        "text": text[:4000],  # Limite pour éviter dépassement contexte
        "max_length": max_length
    })
    
    return result.content

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "docsum"}

@app.post("/v1/docsum/text", response_model=SummaryResponse)
async def summarize_from_text(request: SummarizeTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    try:
        summary = await summarize_text(
            request.text,
            request.style,
            request.max_length
        )
        return SummaryResponse(
            summary=summary,
            original_length=len(request.text.split()),
            summary_length=len(summary.split()),
            language=request.language,
            style=request.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/docsum/file", response_model=SummaryResponse)
async def summarize_from_file(
    file: UploadFile = File(...),
    style: str = "professional",
    max_length: int = 500
):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = extract_text_from_file(tmp_path, ext)
        
        if not text.strip():
            raise HTTPException(status_code=422, detail="Aucun texte extrait.")

        summary = await summarize_text(text, style, max_length)
        
        return SummaryResponse(
            summary=summary,
            original_length=len(text.split()),
            summary_length=len(summary.split()),
            filename=filename,
            language="fr",
            style=style
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)