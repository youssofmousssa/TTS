from pathlib import Path
import os
import tempfile
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq

# ------------------------------
# Replace with your API key (or use env var os.getenv("GROQ_API_KEY"))
# ------------------------------
GROQ_API_KEY = "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"

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="Groq TTS Service")

# Voices lists (from your message)
ENGLISH_VOICES = [
    "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI",
    "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI",
    "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI",
    "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI",
    "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
]

ARABIC_VOICES = [
    "Ahmad-PlayAI", "Amira-PlayAI", "Khalid-PlayAI", "Nasser-PlayAI"
]

# Default voices
DEFAULT_ENGLISH_VOICE = "Arista-PlayAI"
DEFAULT_ARABIC_VOICE = "Ahmad-PlayAI"

# Models
ENGLISH_MODEL = "playai-tts"
ARABIC_MODEL = "playai-tts-arabic"


# Pydantic model for POST requests
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # optional, will pick default based on language param
    lang: Optional[str] = "en"   # "en" or "ar"
    # You can add more options here (format, sample_rate, etc. if supported)


def _validate_and_select_model_voice(lang: str, voice: Optional[str]) -> tuple[str, str]:
    """
    Returns (model, voice) after validation and default-selection.
    Raises HTTPException on invalid input.
    """
    lang = (lang or "en").lower()
    if lang not in ("en", "ar"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'ar'")

    if lang == "ar":
        model = ARABIC_MODEL
        if voice:
            if voice not in ARABIC_VOICES:
                raise HTTPException(status_code=400, detail=f"Arabic voice not found. Use one of: {ARABIC_VOICES}")
            selected_voice = voice
        else:
            selected_voice = DEFAULT_ARABIC_VOICE
    else:
        model = ENGLISH_MODEL
        if voice:
            if voice not in ENGLISH_VOICES:
                raise HTTPException(status_code=400, detail=f"English voice not found. Use one of: {ENGLISH_VOICES}")
            selected_voice = voice
        else:
            selected_voice = DEFAULT_ENGLISH_VOICE

    return model, selected_voice


def _cleanup_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


def _synthesize_to_wav(model: str, voice: str, text: str, out_path: str):
    """
    Calls Groq client to synthesize text -> wav file.
    Uses the same pattern you provided: response.stream_to_file(...)
    """
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        response_format="wav",
        input=text,
    )
    response.stream_to_file(out_path)


@app.get("/voices")
def list_voices():
    return JSONResponse({
        "english": ENGLISH_VOICES,
        "arabic": ARABIC_VOICES,
        "defaults": {"en": DEFAULT_ENGLISH_VOICE, "ar": DEFAULT_ARABIC_VOICE}
    })


@app.get("/tts/")
def tts_get(text: str = "Hello world !!", voice: Optional[str] = None, lang: str = "en", background_tasks: BackgroundTasks = None):
    """
    Generate TTS via GET.
    Example: /tts/?text=Hi+there&voice=Atlas-PlayAI&lang=en
    lang = "en" or "ar"
    """
    model, selected_voice = _validate_and_select_model_voice(lang, voice)

    # create unique temp file per request to avoid collisions
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpfile_path = tmpfile.name
    tmpfile.close()

    try:
        _synthesize_to_wav(model=model, voice=selected_voice, text=text, out_path=tmpfile_path)
    except Exception as e:
        # cleanup on error
        _cleanup_file(tmpfile_path)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    # schedule deletion after response is sent
    background_tasks.add_task(_cleanup_file, tmpfile_path)

    return FileResponse(path=tmpfile_path, media_type="audio/wav", filename="speech.wav")


@app.post("/tts/")
def tts_post(req: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS via POST with JSON body:
    {
      "text": "Hello",
      "voice": "Atlas-PlayAI",   # optional
      "lang": "en"              # optional: "en" or "ar"
    }
    """
    model, selected_voice = _validate_and_select_model_voice(req.lang, req.voice)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpfile_path = tmpfile.name
    tmpfile.close()

    try:
        _synthesize_to_wav(model=model, voice=selected_voice, text=req.text, out_path=tmpfile_path)
    except Exception as e:
        _cleanup_file(tmpfile_path)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    background_tasks.add_task(_cleanup_file, tmpfile_path)
    return FileResponse(path=tmpfile_path, media_type="audio/wav", filename="speech.wav")
