"""Facial expression estimate via FER+ ONNX from Hugging Face (onnxmodelzoo)."""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from app.services.gender_service import (
    _face_roi_from_landmarks,
    _image_bytes_to_bgr,
    _largest_face_roi_bgr,
)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "emotion_onnx"
_MODEL_FILE = _MODEL_DIR / "emotion-ferplus-8.onnx"
_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
)

# FER+ 8-class head (ONNX Model Zoo / paper ordering)
_EMOTION_LABELS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
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
    with urllib.request.urlopen(_MODEL_URL, timeout=300) as r:  # noqa: S310
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


def _face_to_ferplus_input(face_bgr: np.ndarray) -> np.ndarray:
    """NCHW float32 [0,1] grayscale 64×64 — matches common FER+ ONNX deployments."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = (small / 255.0).reshape(1, 1, 64, 64)
    return x


def predict_emotion_from_bytes(
    image_bytes: bytes,
    landmark_points: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Return dominant FER+ emotion + confidence; for UX only — not clinical."""
    try:
        bgr = _image_bytes_to_bgr(image_bytes)
    except Exception as e:
        return {
            "label": "unknown",
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

    inp = _face_to_ferplus_input(face)

    try:
        sess = _session()
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: inp})[0]
    except Exception as e:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "model": "emotion_ferplus_onnx_hf",
            "error": str(e)[:200],
        }

    logits = np.asarray(out).reshape(-1)
    n = logits.size
    if n != len(_EMOTION_LABELS):
        return {
            "label": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "model": "emotion_ferplus_onnx_hf",
            "error": f"Expected {len(_EMOTION_LABELS)} logits, got {n}",
        }

    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -40.0, 40.0))
    probs = (e / max(float(e.sum()), 1e-9)).astype(np.float64)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = _EMOTION_LABELS[idx] if 0 <= idx < len(_EMOTION_LABELS) else "unknown"

    return {
        "label": label,
        "label_index": idx,
        "confidence": round(conf, 3),
        "low_confidence": conf < 0.35,
        "model": "onnxmodelzoo/emotion-ferplus-8 (Hugging Face)",
        "provenance": _MODEL_URL,
    }
