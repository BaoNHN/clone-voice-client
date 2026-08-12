# Hướng Dẫn — clone-voice-client

> **Đây là một thư viện Python (SDK), không phải một ứng dụng chạy được.** Không có `app.py`, không có server, không có port riêng, không có gì để "khởi động". Thư mục này trông "trống" so với 3 project anh em (`rag-legal-assistant`, `clone-voice-station`, `voice-lab-example`) là **chủ đích**, không phải thiếu sót — nó chỉ tồn tại để 2 project kia `import` như một thư viện bình thường.

---

## Mục Lục

1. [Đây Là Gì / Đây KHÔNG Phải Là Gì](#1-đây-là-gì--đây-không-phải-là-gì)
2. [Cài Đặt](#2-cài-đặt)
3. [Cách Dùng Cơ Bản](#3-cách-dùng-cơ-bản)
4. [Chế Độ Local STT (tuỳ chọn)](#4-chế-độ-local-stt-tuỳ-chọn)
5. [Ai Đang Dùng Thư Viện Này](#5-ai-đang-dùng-thư-viện-này)
6. [Tài Liệu API Đầy Đủ](#6-tài-liệu-api-đầy-đủ)
7. [Cấu Trúc Thư Mục](#7-cấu-trúc-thư-mục)
8. [Xử Lý Sự Cố](#8-xử-lý-sự-cố)

---

## 1. Đây Là Gì / Đây KHÔNG Phải Là Gì

`clone-voice-client` là **HTTP client SDK cho `clone-voice-station`** — gói lại toàn bộ lời gọi HTTP tới service giọng nói (STT/TTS/RVC) thành các hàm Python bình thường, để một ứng dụng khách (host app) không phải tự viết lại lớp gọi HTTP đó.

| | |
|---|---|
| ✅ Là gì | Một thư viện Python (`pip install -e`), import và gọi hàm trực tiếp trong tiến trình của app đang dùng nó |
| ❌ Không phải là gì | Một service/server độc lập — không có `app.py`, không lắng nghe port nào, không có terminal riêng để chạy |
| Được dùng bởi | `rag-legal-assistant`, `voice-lab-example` (cài với extra `[local]`), và các script trong `clone-voice-station/tools/` (cài không kèm `[local]`) |
| Bắt buộc nằm ở đâu | Phải là thư mục **anh em cùng cấp** với các project dùng nó — cài qua đường dẫn tương đối `../clone-voice-client`, không phải từ PyPI |

Xem sơ đồ toàn hệ thống tại `clone-voice-station/HUONG_DAN_DEMO.html` (trang tổng hợp cho cả 4 project) để thấy rõ vị trí thư viện này trong kiến trúc chung.

---

## 2. Cài Đặt

Thư viện này **không tự cài** — nó được cài gián tiếp khi bạn `pip install -r requirements.txt` ở một trong các project anh em, nhờ dòng:

```
-e ../clone-voice-client[local]     # rag-legal-assistant, voice-lab-example
-e ../clone-voice-client            # clone-voice-station (chỉ dùng tools/, không cần [local])
```

`-e` (editable install) nghĩa là pip trỏ thẳng vào mã nguồn thư mục này thay vì copy — sửa code ở đây có hiệu lực ngay cho mọi project đang dùng nó, không cần cài lại.

**Yêu cầu cấu trúc thư mục** (bắt buộc, xem thêm `HUONG_DAN_CHAY_TOAN_HE_THONG.md` ở `clone-voice-station`):

```
D:\hoc\project\
├── clone-voice-client\      ← thư mục này
├── clone-voice-station\
├── rag-legal-assistant\
└── voice-lab-example\
```

Thiếu thư mục này ở đúng vị trí anh em → `pip install -r requirements.txt` ở 2 project kia báo lỗi "path not found" ngay bước cài, và cả 2 app đều không chạy được.

**Extra `[local]`** thêm `openai-whisper`, `transformers`, `peft` (kéo theo `torch`) — chỉ cần khi host app muốn chạy STT **ngay trong tiến trình của nó** thay vì gọi qua mạng tới `clone-voice-station` (mục 4). Không cần extra này nếu chỉ gọi API từ xa.

---

## 3. Cách Dùng Cơ Bản

```python
from clone_voice_client import VoiceStationClient

client = VoiceStationClient.from_key_file("voice_station_key.txt")
# hoặc: VoiceStationClient(base_url="http://127.0.0.1:8090", api_key="...")
# hoặc bỏ trống cả 2 tham số: tự đọc từ biến môi trường VOICE_STATION_URL / VOICE_STATION_API_KEY

client.record_voice_consent(external_user_id)
audio = client.speak("Xin chào!", external_user_id)["audio"]
```

- `external_user_id` là một chuỗi bất kỳ do host app tự chọn để định danh người dùng của mình (VD `str(user_id)` trong database riêng) — `clone-voice-station` không có khái niệm tài khoản người dùng thật, chỉ phân biệt theo cặp (API key, `external_user_id`).
- Toàn bộ lời gọi phía sau đều là HTTP request thật tới một `clone-voice-station` đang chạy (mặc định `http://127.0.0.1:8090`) — cần service đó sống thì các hàm này mới trả kết quả đúng (xem `clone-voice-station/terminal.txt`).

---

## 4. Chế Độ Local STT (tuỳ chọn)

Thay vì gọi `client.transcribe(...)` (đi qua mạng tới `clone-voice-station`), có thể chạy Whisper **ngay trong tiến trình của host app**, không gọi mạng — cần cài với extra `[local]`.

```python
# Cách 1: qua VoiceStationClient (tự kiểm tra [local] đã cài chưa)
result = client.transcribe_local(filename, content, mime=mime, language="vi")

# Cách 2: gọi thẳng module local_stt (VoiceStationClient.transcribe_local() cũng chỉ gọi vào đây)
from clone_voice_client import local_stt

result = local_stt.transcribe(audio_bytes, mime="audio/webm", language="vi")
# result = {"text": "...", "language": "vi"}
```

Nếu chưa cài `[local]` mà gọi `transcribe_local()`, lỗi trả về rõ ràng: `"Chế độ local STT chưa được cài — chạy: pip install clone-voice-client[local]"`.

### Dùng STT pack tải từ STT Lab (hotword / LoRA fine-tune)

```python
pack = local_stt.load_pack(zip_path)          # giải nén .stt-pack.zip, đọc manifest.json
result = local_stt.transcribe_with_lora(       # nếu pack là Tier 2 (LoRA)
    audio_bytes, base_model=pack["base_model"], adapter_dir=pack["adapter_dir"]
)
```

`.stt-pack.zip` tải về từ `clone-voice-station`'s trang `/stt-lab` (Tier 1 = hotword bias, Tier 2 = LoRA fine-tune thật) — xem `clone-voice-station/HUONG_DAN_DEMO.html`.

### Model size & ffmpeg

- Biến môi trường `CLONE_VOICE_LOCAL_MODEL` chọn kích cỡ Whisper (`tiny`/`base`/`small`/`medium`, mặc định `small`).
- Biến môi trường `CLONE_VOICE_FFMPEG_DIR` trỏ tới một bản ffmpeg tĩnh nếu ffmpeg hệ thống lỗi (thư viện này **không** đóng gói sẵn ffmpeg riêng — chỉ đọc biến môi trường này nếu host app tự set).

---

## 5. Ai Đang Dùng Thư Viện Này

| Project | Cài kèm `[local]`? | Dùng để làm gì |
|---|---|---|
| `rag-legal-assistant` (`voice/station_client.py`) | Có | Bọc `VoiceStationClient` thành các hàm module-level (`station_client.speak(...)` v.v.) để phần còn lại của app gọi như cũ; dùng `local_stt` khi admin bật chế độ STT local (xem `admin_voice_models.html`) |
| `voice-lab-example` (`app.py`) | Có | Minh hoạ trực tiếp 2 đường transcribe — local (`local_stt`) vs remote (`VoiceStationClient.transcribe()`) — trên cùng một trang `/compare` |
| `clone-voice-station` (`tools/test_stt.py`, `tools/eval_stt_wer.py`) | Không (chỉ cần đường remote) | Script CLI tự kiểm thử/đánh giá WER — gọi vào chính `/api/transcribe` của service này như một client bên ngoài (dogfooding) |

---

## 6. Tài Liệu API Đầy Đủ

Class chính: `VoiceStationClient` (`clone_voice_client/client.py`). Import từ package gốc: `from clone_voice_client import VoiceStationClient, VoiceStationError, MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER`.

### Khởi tạo

| Hàm | Mô tả |
|---|---|
| `VoiceStationClient(base_url=None, api_key=None, *, request_timeout=15, speak_timeout=30, upload_timeout=30)` | `base_url`/`api_key` bỏ trống thì tự đọc biến môi trường `VOICE_STATION_URL` / `VOICE_STATION_API_KEY` |
| `VoiceStationClient.from_key_file(key_path, **kwargs)` | Đọc API key từ file (VD `voice_station_key.txt`, đã gitignore) |
| `.get_own_api_key()` | Trả lại API key đang dùng |
| `.is_available()` | `True`/`False` — ping `/api/health` |

### Kịch bản / đồng ý / hồ sơ giọng nói

| Hàm | Mô tả |
|---|---|
| `.get_scripts()` | Kịch bản đọc mẫu để thu âm |
| `.has_voice_consent(external_user_id)` | Trả `False` nếu không kết nối được station (an toàn, không raise) |
| `.record_voice_consent(external_user_id)` | Ghi nhận đồng ý thu âm |
| `.list_voice_profiles(external_user_id)` | Danh sách giọng (builtin + của người dùng này) |
| `.create_voice_profile(external_user_id, name)` | Tạo hồ sơ giọng mới, trả về `profile_id` |
| `.update_voice_profile(profile_id, external_user_id, name=None, is_default=None)` | Đổi tên / đặt mặc định |
| `.delete_voice_profile(profile_id, external_user_id)` | Xoá giọng của chính người dùng |
| `.get_voice_profile_status(profile_id, external_user_id)` | Trạng thái huấn luyện |

### Mẫu ghi âm & huấn luyện

| Hàm | Mô tả |
|---|---|
| `.upload_voice_sample(profile_id, external_user_id, script_id, filename, content: bytes)` | Upload 1 đoạn ghi âm |
| `.list_voice_samples(profile_id, external_user_id)` | Danh sách mẫu đã ghi |
| `.delete_voice_sample(profile_id, sample_id, external_user_id)` | Xoá 1 mẫu |
| `.train_voice_profile(profile_id, external_user_id)` | Bắt đầu huấn luyện (chạy nền phía station) |

### Nhận diện giọng nói (STT)

| Hàm | Mô tả |
|---|---|
| `.transcribe(filename, content: bytes, mime=None, language="vi")` | Gọi `/api/transcribe` từ xa. Trả `{"text","language","segments"}`. Không cần consent — chỉ là ASR thuần |
| `.transcribe_local(filename, content: bytes, mime=None, language="vi", hotwords=None, pack=None)` | Chạy Whisper ngay trong tiến trình (mục 4), cần extra `[local]`. Trả `{"text","language"}` |

### Đọc to (TTS/RVC)

| Hàm | Mô tả |
|---|---|
| `.speak(text, external_user_id, profile_id=None)` | Trả `{"audio": bytes, "mime": str}` |

### Quản trị (admin API — của client bạn, không phải dashboard manager của station)

| Hàm | Mô tả |
|---|---|
| `.list_all_voice_profiles()` | Mọi giọng riêng của client bạn |
| `.admin_retrain_voice_model(profile_id)` | Huấn luyện lại |
| `.admin_disable_voice_model(profile_id)` | Vô hiệu hoá |
| `.admin_delete_voice_model(profile_id)` | Xoá hẳn |
| `.get_rvc_endpoint()` / `.set_rvc_endpoint(endpoint)` | Xem/đổi URL Colab RVC (ảnh hưởng **mọi** client dùng chung station) |

### Thông báo (webhook/polling)

| Hàm | Mô tả |
|---|---|
| `.register_webhook(webhook_url)` | An toàn gọi lại mỗi lần app khởi động — chỉ là upsert |
| `.poll_undelivered_notifications(external_user_id)` | Thông báo chưa xác nhận (dự phòng khi webhook lỡ thất bại) |
| `.ack_notification(notification_id)` | Xác nhận đã nhận, không gửi lại nữa |

### Hằng số & lỗi

- `MIN_TRAIN_SAMPLES = 5`, `MAX_CLONED_VOICES_PER_USER = 2` — bản sao hiển thị của giới hạn phía `clone-voice-station` (station vẫn tự kiểm tra lại, đây chỉ để host app render UI mà không cần gọi API trước).
- `VoiceStationError(message, status_code=502)` — mọi lỗi giao tiếp với station đều raise lỗi này, `message` đã là text an toàn để hiển thị thẳng cho người dùng.

---

## 7. Cấu Trúc Thư Mục

```
clone-voice-client/
├── clone_voice_client/
│   ├── __init__.py       # export VoiceStationClient, VoiceStationError, MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER
│   ├── client.py          # VoiceStationClient — SDK gọi HTTP tới clone-voice-station
│   └── local_stt.py        # chế độ Local STT — chạy Whisper trong tiến trình (mục 4)
└── pyproject.toml          # tên gói: clone-voice-client, extra "local"
```

Không có `app.py`, không có `examples/`, không có `tests/` — không có gì để chạy độc lập.

---

## 8. Xử Lý Sự Cố

### `pip install -r requirements.txt` báo lỗi liên quan `../clone-voice-client`

**Nguyên nhân:** thư mục `clone-voice-client` không nằm cùng cấp với project đang cài (mục 2). **Khắc phục:** đảm bảo cấu trúc 4 thư mục anh em đúng như mục 2, chạy lại `pip install`.

### Gọi `transcribe_local()` báo "Chế độ local STT chưa được cài"

**Nguyên nhân:** project đang cài `clone-voice-client` **không kèm** extra `[local]` (VD `clone-voice-station/requirements.txt` cố tình không cài, vì nó không cần). **Khắc phục:** nếu thực sự cần STT local ở project đó, sửa `requirements.txt` thành `-e ../clone-voice-client[local]` rồi cài lại.

### Sửa code trong `clone_voice_client/` nhưng project dùng nó không thấy thay đổi

**Kiểm tra:** project đó có cài bằng `pip install -e ...` (editable) không, hay lỡ cài bản đã đóng gói (`pip install clone-voice-client` từ một wheel cũ). Cài lại bằng đúng dòng `-e ../clone-voice-client[local]` trong `requirements.txt` của project đó.

### `VoiceStationError` khi gọi bất kỳ hàm nào (`.transcribe()`, `.speak()`, ...)

**Nguyên nhân:** `clone-voice-station` chưa chạy, hoặc API key sai/hết hạn. **Kiểm tra:** xem `clone-voice-station/terminal.txt` để khởi động service, và xác nhận đúng API key trong file key hoặc biến môi trường `VOICE_STATION_API_KEY`.
