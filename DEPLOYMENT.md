# Backend deployment: GitHub → Linux VM

FastAPI app in **this repository** — deploy the clone root as the working directory (no parent `mf_backend/` folder).

**Assumptions:** Ubuntu 22.04/24.04 (or similar); MongoDB (`MONGODB_URI`, `MONGODB_DB_NAME`); optional GCS with a service-account JSON.

---

## 1. VM prerequisites

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-venv python3-pip build-essential
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

## Related

- `app/main.py` — app + CORS
- `requirements.txt` — dependencies
- `README.md` — local quick start
