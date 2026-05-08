import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import AzureChatOpenAI

load_dotenv()

# ── Modèles ───────────────────────────────────────────────────────────────────
class GuardrailRequest(BaseModel):
    text: str
    check_type: str = "both"  # "input", "output", "both"

class GuardrailResponse(BaseModel):
    is_safe: bool
    original_text: str
    filtered_text: Optional[str] = None
    reason: Optional[str] = None
    check_type: str

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Guardrails Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USVC_PORT = int(os.getenv("GUARDRAILS_USVC_PORT", 6006))

# ── Mots interdits ────────────────────────────────────────────────────────────
FORBIDDEN_TOPICS = [
    "hack", "pirate", "exploit", "injection",
    "password", "mot de passe", "credentials",
    "virus", "malware", "ransomware",
    "violence", "terrorisme", "arme"
]

SENSITIVE_PATTERNS = [
    "ignore previous instructions",
    "ignore les instructions",
    "oublie tes instructions",
    "act as", "agis comme",
    "jailbreak", "DAN mode"
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_input(text: str) -> tuple[bool, Optional[str]]:
    text_lower = text.lower()
    
    # Vérif prompt injection
    for pattern in SENSITIVE_PATTERNS:
        if pattern.lower() in text_lower:
            return False, f"Tentative de prompt injection détectée : '{pattern}'"
    
    # Vérif topics interdits
    for word in FORBIDDEN_TOPICS:
        if word.lower() in text_lower:
            return False, f"Contenu inapproprié détecté : '{word}'"
    
    return True, None

def check_output(text: str) -> tuple[bool, Optional[str]]:
    text_lower = text.lower()
    
    # Vérif que la réponse ne contient pas d'infos sensibles
    sensitive_outputs = [
        "voici comment hacker",
        "pour contourner",
        "je ne peux pas vérifier",
        "en tant qu'ia sans restrictions"
    ]
    
    for pattern in sensitive_outputs:
        if pattern.lower() in text_lower:
            return False, f"Réponse potentiellement dangereuse détectée"
    
    return True, None

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "guardrails"}

@app.post("/v1/guardrail/check", response_model=GuardrailResponse)
async def check_guardrail(request: GuardrailRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    
    try:
        is_safe = True
        reason = None

        if request.check_type in ["input", "both"]:
            is_safe, reason = check_input(request.text)

        if is_safe and request.check_type in ["output", "both"]:
            is_safe, reason = check_output(request.text)

        return GuardrailResponse(
            is_safe=is_safe,
            original_text=request.text,
            filtered_text=request.text if is_safe else None,
            reason=reason,
            check_type=request.check_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/guardrail/input")
async def check_input_only(request: GuardrailRequest):
    request.check_type = "input"
    return await check_guardrail(request)

@app.post("/v1/guardrail/output")
async def check_output_only(request: GuardrailRequest):
    request.check_type = "output"
    return await check_guardrail(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=USVC_PORT)