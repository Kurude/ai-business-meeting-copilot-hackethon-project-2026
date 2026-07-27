"""
Voice service:
- Speech-to-Text via Gemini's native audio understanding (the audio is sent straight to
  the same gemini-2.5-flash model used for chat, with a transcription-only prompt) -
  no separate STT provider or API key needed beyond GEMINI_API_KEY.
- Text-to-Speech via gTTS (Google Text-to-Speech) - free, no API key required, and
  unrelated to the Gemini API key.
"""
import os
import uuid
import mimetypes

from gtts import gTTS
from google.genai import types

from llm_service import get_client, CHAT_MODEL

AUDIO_OUT_DIR = os.path.join(os.path.dirname(__file__), "uploads", "tts")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

_EXT_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}

_TRANSCRIBE_PROMPT = (
    "Transcribe this audio verbatim. Return ONLY the spoken words as plain text - "
    "no timestamps, no speaker labels, no markdown, no commentary. "
    "If there is no discernible speech, return an empty string."
)


def transcribe_audio(file_path: str) -> str:
    """Send an audio file directly to Gemini and return the transcript text."""
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = _EXT_MIME_MAP.get(ext) or mimetypes.guess_type(file_path)[0] or "audio/webm"

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    if not audio_bytes:
        return ""

    client = get_client()
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            _TRANSCRIBE_PROMPT,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )
    return (response.text or "").strip()


def text_to_speech(text: str) -> str:
    """Convert text to an mp3 file and return its path."""
    filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join(AUDIO_OUT_DIR, filename)
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)
    return output_path
