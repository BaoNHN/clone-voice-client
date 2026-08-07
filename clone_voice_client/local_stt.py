"""
clone_voice_client/local_stt.py
Runs Whisper transcription in this host app's own process — the "local" mode
a host app opts into via `pip install clone-voice-client[local]`, as opposed
to VoiceStationClient.transcribe()'s HTTP call to clone-voice-station's
/api/transcribe. Mirrors clone-voice-station's own voice/stt.py lazy-load
pattern (model loaded on first use, not at import time) since the two repos
can't share code directly.

All whisper/torch imports stay inside _load_model() so importing this module
(or clone_voice_client generally) never requires the [local] extra unless a
caller actually invokes transcribe().
"""

import json
import os
import tempfile
import zipfile

_model = None
_MODEL_SIZE = os.getenv("CLONE_VOICE_LOCAL_MODEL", "small")  # tiny | base | small | medium

# A host app can point this at its own bundled ffmpeg directory if the platform's
# default ffmpeg doesn't launch cleanly (see clone-voice-station's voice/stt.py
# for the Windows conda-forge/ffmpeg DLL-conflict this works around) — this SDK
# ships no bundled binary of its own, just the opt-in hook.
_FFMPEG_DIR = os.getenv("CLONE_VOICE_FFMPEG_DIR")
if _FFMPEG_DIR and os.path.isdir(_FFMPEG_DIR):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

_SUFFIX_BY_MIME = {
    "webm": ".webm",
    "ogg":  ".ogg",
    "mp4":  ".m4a",
    "m4a":  ".m4a",
    "mpeg": ".mp3",
    "mp3":  ".mp3",
    "wav":  ".wav",
}


def _load_model():
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(_MODEL_SIZE)
    return _model


def _suffix_for(mime: str) -> str:
    mime = (mime or "").lower()
    for key, suffix in _SUFFIX_BY_MIME.items():
        if key in mime:
            return suffix
    return ".webm"  # MediaRecorder's default container when nothing else matches


def transcribe(audio_bytes: bytes, mime: str = "audio/webm", language: str = None,
                initial_prompt: str = None) -> dict:
    """
    Parameters
    ----------
    audio_bytes    : bytes  Raw audio, typically from a browser MediaRecorder.
    mime           : str    MIME type hint, used only to pick a temp-file suffix.
    language       : str    ISO 639-1 code (e.g. "vi") to force a language, or
                             None to let Whisper auto-detect.
    initial_prompt : str    Vocabulary-biasing hint passed straight through to
                             Whisper — this is how a downloaded STT Lab hotword
                             list (see load_hotwords_from_pack()) actually
                             influences local transcription.

    Returns
    -------
    dict {"text": str, "language": str}
    """
    model = _load_model()

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(mime), delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = model.transcribe(
            tmp_path,
            language=language or None,
            task="transcribe",
            fp16=False,  # CPU-safe; harmless no-op on CUDA where fp16 is autodetected anyway
            initial_prompt=initial_prompt or None,
        )
        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
        }
    finally:
        os.unlink(tmp_path)


def load_hotwords_from_pack(zip_path: str) -> list:
    """Reads hotwords.json out of a .stt-pack.zip downloaded from
    clone-voice-station's STT Lab (GET /api/stt/adapters/{id}/download)."""
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("hotwords.json"))
