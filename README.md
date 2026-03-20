# mf_backend

FastAPI backend for virtual try-on: glasses detection/removal, iris-based landmarks & PD, optional Hugging Face ONNX PD, gender/age estimates, eyewear merchandising hints, MongoDB session storage, and optional GCS uploads.

## Quick run (local)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # create and edit .env (see below)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment

Create `.env` in this directory (not committed). Required fields match `app/utils/settings.py`:

- `MONGODB_URI`, `MONGODB_DB_NAME`
- `GOOGLE_APPLICATION_CREDENTIALS`, `GCP_PROJECT_ID`, `BUCKET_NAME`, `FOLDER_NAME` (if using GCS)
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` (optional)

Place your GCS service-account JSON under `secrets/` (gitignored) and set `GOOGLE_APPLICATION_CREDENTIALS` to its absolute path.

## ONNX / models

Weights under `app/models/*_onnx/` are gitignored; many download on first API use. Larger assets (e.g. glasses detector `.pth`, MediaPipe `.task`) may already be in the repo or restored from your backup.

## Deployment

See **[DEPLOYMENT.md](./DEPLOYMENT.md)** for GitHub → Linux VM (systemd, Nginx, checklist).

## API

- `GET /` — health
- Routers: glasses, landmarks, virtual try-on (see `app/routes/`)
