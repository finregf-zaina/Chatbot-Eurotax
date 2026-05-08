import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil

# PDF
from pypdf import PdfReader

# DOCX  
import docx2txt

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class ExtractResponse(BaseModel):
    filename: str
    text: str
    pages: int
    file_type: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Text Extractor Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("TEXT_EXTRACTOR_USVC_PORT", 6000))

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf(file_path: str) -> tuple[str, int]:
    reader = PdfReader(file_path)
    pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text, pages

def extract_docx(file_path: str) -> tuple[str, int]:
    text = docx2txt.process(file_path)
    return text, 1

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "text_extractor"}

@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Formats acceptés : .pdf, .docx"
        )

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            text, pages = extract_pdf(tmp_path)
            file_type = "pdf"
        else:
            text, pages = extract_docx(tmp_path)
            file_type = "docx"

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Aucun texte extrait du document."
            )

        return ExtractResponse(
            filename=filename,
            text=text,
            pages=pages,
            file_type=file_type
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