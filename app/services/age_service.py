"""Age band estimate via Levi / Adience GoogLeNet ONNX from Hugging Face (onnxmodelzoo)."""

from __future__ import annotations

import io
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from app.services.gender_service import (
    _face_roi_from_landmarks,
    _largest_face_roi_bgr,
)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "age_onnx"
_MODEL_FILE = _MODEL_DIR / "age_googlenet.onnx"
_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/age_googlenet/resolve/main/age_googlenet.onnx"
)

# Levi et al. 8-way Adience buckets (years) — matches OpenCV / model zoo convention
_AGE_BUCKETS = (
    "(0, 2)",
    "(4, 6)",
    "(8, 12)",
    "(15, 20)",
    "(25, 32)",
    "(38, 43)",
    "(48, 53)",
    "(60, 100)",
)

_LOCK = threading.Lock()
_SESSION: Optional[ort.InferenceSession] = None


def _ensure_model() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _MODEL_FILE.exists():
        sz = _MODEL_FILE.stat().st_size
        if sz >= 1_000_000:
            with open(_MODEL_FILE, "rb") as fh:
                head = fh.read(40)
            if head.startswith(b"version https://git-lfs"):
                try:
                    _MODEL_FILE.unlink()
                except OSError:
                    pass
            else:
                return
        else:
            try:
                _MODEL_FILE.unlink()
            except OSError:
                pass
    with urllib.request.urlopen(_MODEL_URL, timeout=180) as r:  # noqa: S310
        _MODEL_FILE.write_bytes(r.read())


def _session() -> ort.InferenceSession:
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            _ensure_model()
            try:
                _SESSION = ort.InferenceSession(
                    str(_MODEL_FILE), providers=["CPUExecutionProvider"]
                )
            except Exception:
                try:
                    _MODEL_FILE.unlink()
                except OSError:
                    pass
                _ensure_model()
                _SESSION = ort.InferenceSession(
                    str(_MODEL_FILE), providers=["CPUExecutionProvider"]
                )
        return _SESSION


def _image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR (shared semantics with gender_service)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        rgba = img.mode in ("RGBA", "LA")
        img = img.convert("RGBA" if rgba else "RGB")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            rgb = arr[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            blended = (alpha * rgb + (1.0 - alpha) * white).astype(np.uint8)
            return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError("Could not decode image for age estimation")
        if bgr.ndim == 2:
            return cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        return bgr


def predict_age_from_bytes(
    image_bytes: bytes,
    landmark_points: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Return age bucket + confidence; for UX / merchandising only — not identity or Rx."""
    try:
        bgr = _image_bytes_to_bgr(image_bytes)
    except Exception as e:
        return {
            "bucket": "unknown",
            "bucket_index": None,
            "confidence": 0.0,
            "low_confidence": True,
            "model": None,
            "error": str(e)[:200],
        }

    try:
        if landmark_points and len(landmark_points) >= 468:
            face = _face_roi_from_landmarks(bgr, landmark_points)
        else:
            face = _largest_face_roi_bgr(bgr)
    except Exception:
        face = _largest_face_roi_bgr(bgr)

    face = cv2.resize(face, (224, 224))
    blob = cv2.dnn.blobFromImage(
        face, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=False
    )

    try:
        sess = _session()
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: blob})[0]
    except Exception as e:
        return {
            "bucket": "unknown",
            "bucket_index": None,
            "confidence": 0.0,
            "low_confidence": True,
            "model": "age_googlenet_onnx_hf",
            "error": str(e)[:200],
        }

    logits = np.asarray(out).reshape(-1)
    n = logits.size
    if n != len(_AGE_BUCKETS):
        return {
            "bucket": "unknown",
            "bucket_index": None,
            "confidence": 0.0,
            "low_confidence": True,
            "model": "age_googlenet_onnx_hf",
            "error": f"Expected {len(_AGE_BUCKETS)} logits, got {n}",
        }

    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -40.0, 40.0))
    probs = (e / max(float(e.sum()), 1e-9)).astype(np.float64)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    bucket = _AGE_BUCKETS[idx] if 0 <= idx < len(_AGE_BUCKETS) else "unknown"

    return {
        "bucket": bucket,
        "bucket_index": idx,
        "confidence": round(conf, 3),
        "low_confidence": conf < 0.35,
        "model": "onnxmodelzoo/age_googlenet (Hugging Face)",
        "provenance": _MODEL_URL,
    }
