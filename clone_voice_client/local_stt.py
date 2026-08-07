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


def load_pack(zip_path: str) -> dict:
    """Extracts a .stt-pack.zip downloaded from clone-voice-station's STT Lab
    (GET /api/stt/adapters/{id}/download) into a sibling "<zip_path>_extracted"
    directory and reads its manifest.

    Returns {"tier": int, "base_model": str, "hotwords": list[str],
    "adapter_dir": str | None}. adapter_dir is only set for a Tier 2 pack
    (contains adapter_model.safetensors + adapter_config.json from a real
    LoRA fine-tune) — a Tier 1 (hotword-only) pack leaves it None, and callers
    should fall back to plain Whisper + hotword prompt-bias (see transcribe()).
    """
    extract_dir = f"{zip_path}_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    with open(os.path.join(extract_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(extract_dir, "hotwords.json"), encoding="utf-8") as f:
        hotwords = json.load(f)

    has_lora = os.path.exists(os.path.join(extract_dir, "adapter_model.safetensors"))
    return {
        "tier": manifest.get("tier", 1),
        "base_model": manifest.get("base_model"),
        "hotwords": hotwords,
        "adapter_dir": extract_dir if has_lora else None,
    }


_HF_MODEL_BY_NAME = {"whisper-tiny": "openai/whisper-tiny", "whisper-base": "openai/whisper-base"}
_lora_models = {}  # adapter_dir -> (processor, model, device), lazy-loaded per pack


def _load_lora_model(base_model: str, adapter_dir: str):
    if adapter_dir not in _lora_models:
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        hf_model_name = _HF_MODEL_BY_NAME.get(base_model)
        if not hf_model_name:
            raise ValueError(f"Unsupported base_model for LoRA inference: {base_model!r}")

        processor = WhisperProcessor.from_pretrained(hf_model_name, language="vietnamese", task="transcribe")
        base = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
        model = PeftModel.from_pretrained(base, adapter_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _lora_models[adapter_dir] = (processor, model, device)
    return _lora_models[adapter_dir]


def transcribe_with_lora(audio_bytes: bytes, base_model: str, adapter_dir: str,
                          mime: str = "audio/webm", language: str = "vi") -> dict:
    """Runs inference through a Tier 2 LoRA-fine-tuned adapter (transformers +
    peft, not the openai-whisper package transcribe() above uses — PEFT only
    attaches to the HF model class). Requires `pip install clone-voice-client[local]`
    (which now also pulls in transformers + peft, see pyproject.toml)."""
    import librosa
    import torch

    processor, model, device = _load_lora_model(base_model, adapter_dir)

    with tempfile.NamedTemporaryFile(suffix=_suffix_for(mime), delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        audio, _sr = librosa.load(tmp_path, sr=16000, mono=True)
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"text": text.strip(), "language": language or "vi"}
    finally:
        os.unlink(tmp_path)
