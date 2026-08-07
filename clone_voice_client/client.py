"""
clone_voice_client/client.py
HTTP client SDK for clone-voice-station — the standalone service that owns all
voice (STT/TTS/RVC) training, storage, and management for any AI-assistant
project that wants a voice loop without embedding that complexity itself.

Usage
-----
    from clone_voice_client import VoiceStationClient

    client = VoiceStationClient.from_key_file("voice_station_key.txt")
    # or: VoiceStationClient(base_url="http://127.0.0.1:8090", api_key="...")

    client.record_voice_consent(external_user_id)
    audio = client.speak("Xin chào!", external_user_id)["audio"]

A host app authenticates itself to the station with a shared API key (issued
by the station's manager dashboard, see clone-voice-station's README/demo
guide) and identifies each of ITS end users as an opaque external_user_id —
the station has no login of its own and trusts the host app to have already
checked who's logged in / who's an admin. Ownership of a profile is enforced
station-side by (client_id inferred from the API key, external_user_id).

Every method degrades gracefully when the station is unreachable: reads
return empty/None, writes raise VoiceStationError with a message that's safe
to show directly to an end user. Catch that in your own route handlers and
turn it into a clean error response instead of a 500.
"""

import os

import requests

# Display-only copies of clone-voice-station's own limits (it enforces these
# server-side regardless) — handy for rendering hint text client-side without
# an extra round-trip. Keep in sync with clone-voice-station/database/database.py.
MIN_TRAIN_SAMPLES = 5
MAX_CLONED_VOICES_PER_USER = 2


class VoiceStationError(Exception):
    """Raised on any failure talking to clone-voice-station — message is
    already safe, human-readable text, ready to bubble up in an API response."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class VoiceStationClient:
    def __init__(self, base_url: str = None, api_key: str = None, *,
                 request_timeout: int = 15, speak_timeout: int = 30, upload_timeout: int = 30):
        """
        Parameters
        ----------
        base_url : str  Where clone-voice-station is running. Defaults to the
                         VOICE_STATION_URL env var, or http://127.0.0.1:8090.
        api_key  : str  This host app's API key. Defaults to the
                         VOICE_STATION_API_KEY env var. Use from_key_file()
                         instead if you keep the key in a gitignored file.
        """
        self.base_url = (base_url or os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090")).rstrip("/")
        self._api_key = api_key or os.getenv("VOICE_STATION_API_KEY", "")
        self.request_timeout = request_timeout
        self.speak_timeout   = speak_timeout
        self.upload_timeout  = upload_timeout

    @classmethod
    def from_key_file(cls, key_path: str, **kwargs) -> "VoiceStationClient":
        """Reads the API key from a file on disk (e.g. voice_station_key.txt,
        gitignored) — matches how clone-voice-station's manager dashboard
        expects you to store the key it issues you. Any other constructor
        kwarg (base_url, timeouts, ...) passes through unchanged."""
        api_key = ""
        if os.path.exists(key_path):
            with open(key_path) as f:
                api_key = f.read().strip()
        return cls(api_key=api_key, **kwargs)

    def get_own_api_key(self) -> str:
        """This client's own API key — also useful to verify an inbound
        webhook call really came from the station, since it echoes this same
        key back in X-Api-Key (see register_webhook())."""
        return self._api_key

    # ── internal request helpers ────────────────────────────────────────────
    def _headers(self) -> dict:
        return {"X-Api-Key": self._api_key}

    def _raise_for_response(self, resp):
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", f"HTTP {resp.status_code}")
            except Exception:
                detail = f"HTTP {resp.status_code}"
            raise VoiceStationError(detail, status_code=resp.status_code)

    def _get(self, path: str, params: dict = None, timeout: int = None):
        try:
            resp = requests.get(f"{self.base_url}{path}", headers=self._headers(),
                                 params=params, timeout=timeout or self.request_timeout)
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def _post(self, path: str, json: dict = None, timeout: int = None):
        try:
            resp = requests.post(f"{self.base_url}{path}", headers=self._headers(),
                                  json=json or {}, timeout=timeout or self.request_timeout)
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def _put(self, path: str, json: dict = None, timeout: int = None):
        try:
            resp = requests.put(f"{self.base_url}{path}", headers=self._headers(),
                                 json=json or {}, timeout=timeout or self.request_timeout)
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def _delete(self, path: str, params: dict = None, timeout: int = None):
        try:
            resp = requests.delete(f"{self.base_url}{path}", headers=self._headers(),
                                    params=params, timeout=timeout or self.request_timeout)
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Scripts / consent / profiles ────────────────────────────────────────
    def get_scripts(self) -> list:
        return self._get("/api/scripts")

    def has_voice_consent(self, external_user_id: str) -> bool:
        try:
            return bool(self._get("/api/consent", params={"external_user_id": external_user_id}).get("consent"))
        except VoiceStationError:
            return False  # station unreachable — degrade to "no consent" rather than break the caller

    def record_voice_consent(self, external_user_id: str):
        self._post("/api/consent", {"external_user_id": external_user_id})

    def list_voice_profiles(self, external_user_id: str) -> list:
        try:
            return self._get("/api/profiles", params={"external_user_id": external_user_id})
        except VoiceStationError:
            return []

    def create_voice_profile(self, external_user_id: str, name: str) -> int:
        return self._post("/api/profiles", {"external_user_id": external_user_id, "name": name})["profile_id"]

    def update_voice_profile(self, profile_id: int, external_user_id: str,
                              name: str = None, is_default: bool = None):
        payload = {"external_user_id": external_user_id}
        if name is not None:
            payload["name"] = name
        if is_default is not None:
            payload["is_default"] = is_default
        return self._put(f"/api/profiles/{profile_id}", payload)

    def delete_voice_profile(self, profile_id: int, external_user_id: str) -> dict:
        return self._delete(f"/api/profiles/{profile_id}", params={"external_user_id": external_user_id})

    def get_voice_profile_status(self, profile_id: int, external_user_id: str) -> dict:
        return self._get(f"/api/profiles/{profile_id}/status", params={"external_user_id": external_user_id})

    # ── Samples ──────────────────────────────────────────────────────────────
    def upload_voice_sample(self, profile_id: int, external_user_id: str, script_id: str,
                             filename: str, content: bytes) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/api/profiles/{profile_id}/samples",
                headers=self._headers(),
                data={"external_user_id": external_user_id, "script_id": script_id},
                files={"audio": (filename, content)},
                timeout=self.upload_timeout,
            )
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def list_voice_samples(self, profile_id: int, external_user_id: str) -> list:
        return self._get(f"/api/profiles/{profile_id}/samples", params={"external_user_id": external_user_id})

    def delete_voice_sample(self, profile_id: int, sample_id: int, external_user_id: str):
        return self._delete(f"/api/profiles/{profile_id}/samples/{sample_id}",
                             params={"external_user_id": external_user_id})

    def train_voice_profile(self, profile_id: int, external_user_id: str) -> dict:
        return self._post(f"/api/profiles/{profile_id}/train", {"external_user_id": external_user_id})

    # ── Transcribe (STT — input half of the voice loop) ─────────────────────
    def transcribe(self, filename: str, content: bytes, mime: str = None, language: str = "vi") -> dict:
        """Sends recorded microphone audio to clone-voice-station's Whisper/
        PhoWhisper-backed /api/transcribe and returns
        {"text": str, "language": str, "segments": list}. Transcription isn't
        tied to any voice profile/consent — it's just ASR."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/transcribe",
                headers=self._headers(),
                data={"language": language} if language else {},
                files={"audio": (filename, content, mime or "application/octet-stream")},
                timeout=self.upload_timeout,
            )
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return resp.json()

    def transcribe_local(self, filename: str, content: bytes, mime: str = None,
                          language: str = "vi", hotwords: list = None) -> dict:
        """Runs STT in this process (clone_voice_client/local_stt.py) instead of
        calling this station over HTTP — requires `pip install clone-voice-client[local]`.
        `hotwords` (e.g. loaded via local_stt.load_hotwords_from_pack() from a
        .stt-pack.zip downloaded from the station's STT Lab) bias transcription
        via Whisper's initial_prompt. Returns {"text": str, "language": str}."""
        try:
            from . import local_stt
        except ImportError as e:
            raise VoiceStationError(
                "Chế độ local STT chưa được cài — chạy: pip install clone-voice-client[local]",
                status_code=500,
            ) from e
        initial_prompt = ", ".join(hotwords) if hotwords else None
        return local_stt.transcribe(content, mime=mime or "audio/webm", language=language,
                                     initial_prompt=initial_prompt)

    # ── Speak (TTS + optional RVC) ───────────────────────────────────────────
    def speak(self, text: str, external_user_id: str, profile_id: int = None) -> dict:
        """Returns {"audio": bytes, "mime": str}."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/speak",
                headers=self._headers(),
                json={"text": text, "external_user_id": external_user_id, "profile_id": profile_id},
                timeout=self.speak_timeout,
            )
        except requests.exceptions.RequestException as e:
            raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
        self._raise_for_response(resp)
        return {"audio": resp.content, "mime": resp.headers.get("content-type", "audio/mpeg")}

    # ── Admin (client-wide — gate this behind your own app's admin check) ───
    def list_all_voice_profiles(self) -> list:
        return self._get("/api/admin/voice_models")

    def admin_retrain_voice_model(self, profile_id: int) -> dict:
        return self._post(f"/api/admin/voice_models/{profile_id}/retrain")

    def admin_disable_voice_model(self, profile_id: int) -> dict:
        return self._post(f"/api/admin/voice_models/{profile_id}/disable")

    def admin_delete_voice_model(self, profile_id: int) -> dict:
        return self._delete(f"/api/admin/voice_models/{profile_id}")

    def get_rvc_endpoint(self) -> dict:
        return self._get("/api/rvc_endpoint")

    def set_rvc_endpoint(self, endpoint: str) -> dict:
        return self._post("/api/rvc_endpoint", {"endpoint": endpoint})

    # ── Notifications: manager-triggered delete/disable events ─────────────
    def register_webhook(self, webhook_url: str):
        """Self-registers this host app's callback URL so clone-voice-station
        can push delete/disable notifications instead of you having to poll
        for them. Safe to call on every startup — it's just an upsert."""
        self._post("/api/webhook", {"webhook_url": webhook_url})

    def poll_undelivered_notifications(self, external_user_id: str) -> list:
        """Fallback path for when the webhook was never registered or a push
        attempt failed — returns notifications the station couldn't deliver yet."""
        try:
            return self._get("/api/notifications", params={"external_user_id": external_user_id})
        except VoiceStationError:
            return []

    def ack_notification(self, notification_id: int):
        try:
            self._post(f"/api/notifications/{notification_id}/ack")
        except VoiceStationError:
            pass  # best-effort — a missed ack just means it's re-delivered next poll
