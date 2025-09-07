from pathlib import Path
import os
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq

# ------------------------------
# Use env var in production (recommended)
# ------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="Groq TTS Service (fixed streaming)")

# Voices lists
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

DEFAULT_ENGLISH_VOICE = "Arista-PlayAI"
DEFAULT_ARABIC_VOICE = "Ahmad-PlayAI"
ENGLISH_MODEL = "playai-tts"
ARABIC_MODEL = "playai-tts-arabic"


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[str] = "en"


def _validate_and_select_model_voice(lang: str, voice: Optional[str]) -> tuple[str, str]:
    lang = (lang or "en").lower()
    if lang not in ("en", "ar"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'ar'")

    if lang == "ar":
        model = ARABIC_MODEL
        selected_voice = voice if voice else DEFAULT_ARABIC_VOICE
        if selected_voice not in ARABIC_VOICES:
            raise HTTPException(status_code=400, detail=f"Arabic voice not found. Use one of: {ARABIC_VOICES}")
    else:
        model = ENGLISH_MODEL
        selected_voice = voice if voice else DEFAULT_ENGLISH_VOICE
        if selected_voice not in ENGLISH_VOICES:
            raise HTTPException(status_code=400, detail=f"English voice not found. Use one of: {ENGLISH_VOICES}")

    return model, selected_voice


def _cleanup_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


def _synthesize_to_wav(model: str, voice: str, text: str, out_path: str):
    """
    Use the streaming response helper from the Groq SDK so we can call response.stream_to_file(...)
    If streaming helper isn't available for some SDK versions, attempt a fallback to raw bytes.
    """
    try:
        # Preferred: use streaming context manager
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            response_format="wav",
            input=text,
        ) as response:
            # This has the stream_to_file helper on the streaming response object
            response.stream_to_file(file=out_path)
            return
    except AttributeError:
        # The SDK version might not have the streaming helper; fall back below
        pass
    except Exception as e:
        # re-raise as HTTP-friendly error in caller
        raise

    # Fallback: attempt non-streaming call and write raw bytes if present
    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            response_format="wav",
            input=text,
        )
        # try common attributes where bytes may live
        if hasattr(resp, "read"):
            data = resp.read()
        elif hasattr(resp, "content"):
            data = resp.content
        elif hasattr(resp, "raw") and hasattr(resp.raw, "read"):
            data = resp.raw.read()
        else:
            raise RuntimeError("Unexpected binary response object from Groq SDK (no stream/read/content available).")
        # write bytes to file
        with open(out_path, "wb") as f:
            f.write(data)
    except Exception as e:
        # bubble up to caller to handle cleanup and HTTPException
        raise


@app.get("/voices")
def list_voices():
    return JSONResponse({
        "english": ENGLISH_VOICES,
        "arabic": ARABIC_VOICES,
        "defaults": {"en": DEFAULT_ENGLISH_VOICE, "ar": DEFAULT_ARABIC_VOICE}
    })


@app.get("/tts/")
def tts_get(text: str = "Hello world !!", voice: Optional[str] = None, lang: str = "en", background_tasks: BackgroundTasks = None):
    model, selected_voice = _validate_and_select_model_voice(lang, voice)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmpfile_path = tmpfile.name
    tmpfile.close()

    try:
        _synthesize_to_wav(model=model, voice=selected_voice, text=text, out_path=tmpfile_path)
    except Exception as e:
        _cleanup_file(tmpfile_path)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    if background_tasks:
        background_tasks.add_task(_cleanup_file, tmpfile_path)

    return FileResponse(path=tmpfile_path, media_type="audio/wav", filename="speech.wav")


@app.post("/tts/")
def tts_post(req: TTSRequest, background_tasks: BackgroundTasks):
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
