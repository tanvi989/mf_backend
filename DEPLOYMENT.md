# Backend deployment: GitHub → Linux VM

FastAPI app in **this repository** — deploy the clone root as the working directory (no parent `mf_backend/` folder).

**Assumptions:** Ubuntu 22.04/24.04 (or similar); MongoDB (`MONGODB_URI`, `MONGODB_DB_NAME`); optional GCS with a service-account JSON.

---

## 1. VM prerequisites

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential
```

**MediaPipe / OpenCV on a headless Linux VM** still expect basic **EGL/GLES and GL** shared libraries. Without them, `/landmarks/detect` can fail with:

`libGLESv2.so.2: cannot open shared object file: No such file or directory`

Install Mesa userspace libs (Ubuntu/Debian):

```bash
sudo apt install -y libgl1 libglib2.0-0 libegl1 libgles2
# If libgles2 is not found on your release, use:
# sudo apt install -y libgl1-mesa-glx libegl1-mesa libgles2-mesa
```

Use **Python 3.10+** (3.11+ recommended). Prefer **≥2 vCPU / 4 GB RAM** for PyTorch + OpenCV + MediaPipe. Stack targets **CPU** ONNX by default.

---

## 2. Clone this repo

```bash
cd /opt   # or $HOME
git clone https://github.com/tanvi989/mf_backend.git
cd mf_backend
```

---

## 3. Virtualenv & dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

ONNX caches under `app/models/*_onnx/` are gitignored; first API calls may **download** weights. Ensure HTTPS egress and ~1–2 GB free disk.

**Process must own `app/models`:** Uvicorn (e.g. `www-data`, `deploy`) needs **write** access to create:

- `app/models/pd_hf_onnx/` — Hugging Face PD ONNX  
- `app/models/gender_onnx/` — gender ONNX  
- `app/models/age_onnx/` — age ONNX  
- `app/models/emotion_onnx/` — FER+ emotion ONNX (Hugging Face)  
- `app/models/face_landmarker/` — MediaPipe task (if not pre-seeded)

If the app was cloned or `pip install` run as **root**, directories may be root-owned and you’ll see:

`[Errno 13] Permission denied: '.../app/models/pd_hf_onnx'` (and similar for `gender_onnx`, `age_onnx`, `emotion_onnx`).

Fix (replace `deploy` with the user that runs `uvicorn`, e.g. `www-data`):

```bash
cd /var/www/mf_backend   # or your install path
sudo mkdir -p app/models/pd_hf_onnx app/models/gender_onnx app/models/age_onnx app/models/emotion_onnx app/models/face_landmarker
sudo chown -R deploy:deploy app/models
sudo chmod -R u+rwX app/models
# Or make the whole repo owned by that user:
# sudo chown -R deploy:deploy /var/www/mf_backend
```

Restart the API service after fixing ownership.

**PD trace in server logs (optional):** set `PD_TRACE_PRINT=1` in the service environment to dump the same step-by-step PD JSON to stdout on every `/landmarks/detect` (verbose).

---

## 4. `.env` on the server

Copy `.env.example` → `.env` and fill in real values. See `app/utils/settings.py` for the schema.

GCS key: keep JSON **outside git** (e.g. `~/secrets/gcs-service-account.json`), set `GOOGLE_APPLICATION_CREDENTIALS` to an **absolute path**.

---

## 5. Run (smoke test)

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/
```

---

## 6. systemd + Nginx

Use `WorkingDirectory=/opt/mf_backend` (match your clone path). Example unit:

```ini
[Service]
WorkingDirectory=/opt/mf_backend
ExecStart=/opt/mf_backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Start with **`--workers 1`** for ML endpoints unless load-tested otherwise.

Nginx: `proxy_pass http://127.0.0.1:8000;` and `client_max_body_size 25M;` for image uploads.

---

## 7. Frontend & CORS

Point the SPA API base URL at your public API. Tighten `allow_origins` in `app/main.py` for production.

---

## 8. Troubleshooting

| Symptom | Fix |
|--------|-----|
| API returns `libGLESv2.so.2` / `libGL` missing | Install packages in **§1** (`libegl1`, `libgles2`, `libgl1`), restart Uvicorn. |
| `libGL.so.1` not found | `sudo apt install -y libgl1` |
| Landmark errors only on production VM | Minimal Docker/VM images omit Mesa; install the same libs inside the container/VM. |
| `Permission denied` under `app/models/*_onnx` | Service user cannot write model caches — see **§3** (`chown` / `mkdir` / `chmod`). |

---

## Related

- `app/main.py` — app + CORS
- `requirements.txt` — dependencies
- `README.md` — local quick start
