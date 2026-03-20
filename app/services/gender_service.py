import io
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from typing import List, Optional, Tuple

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gender_onnx"
_MODEL_FILE = _MODEL_DIR / "gender_googlenet.onnx"
_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/"
    "age_gender/models/gender_googlenet.onnx"
)

_FAIRFACE_MODEL_FILE = _MODEL_DIR / "fairface_gender_quantized.onnx"
# Hugging Face: onnx-community/fairface_gender_image_detection-ONNX
_FAIRFACE_MODEL_URL = (
    "https://huggingface.co/onnx-community/fairface_gender_image_detection-ONNX/"
    "resolve/main/onnx/model_quantized.onnx"
)

_LOCK = threading.Lock()
_SESSION: Optional[ort.InferenceSession] = None

# Levi/Adience GoogLeNet gender head matches OpenCV sample: class 0 = Male, 1 = Female
_GENDER_LABELS = ("Male", "Female")

_SESSION2: Optional[ort.InferenceSession] = None
# FairFace gender config: id2label {0: "Female", 1: "Male"}
_GENDER_LABELS2 = ("Female", "Male")


def _ensure_model() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _MODEL_FILE.exists() and _MODEL_FILE.stat().st_size > 1_000_000:
        return
    with urllib.request.urlopen(_MODEL_URL, timeout=120) as r:
        _MODEL_FILE.write_bytes(r.read())


def _session() -> ort.InferenceSession:
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            _ensure_model()
            _SESSION = ort.InferenceSession(
                str(_MODEL_FILE), providers=["CPUExecutionProvider"]
            )
        return _SESSION


def _ensure_fairface_model() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _FAIRFACE_MODEL_FILE.exists() and _FAIRFACE_MODEL_FILE.stat().st_size > 1_000_000:
        return
    with urllib.request.urlopen(_FAIRFACE_MODEL_URL, timeout=180) as r:  # noqa: S310
        _FAIRFACE_MODEL_FILE.write_bytes(r.read())


def _session2() -> ort.InferenceSession:
    global _SESSION2
    with _LOCK:
        if _SESSION2 is None:
            _ensure_fairface_model()
            _SESSION2 = ort.InferenceSession(
                str(_FAIRFACE_MODEL_FILE), providers=["CPUExecutionProvider"]
            )
        return _SESSION2


def _predict_gender_fairface_onnx(face_bgr_224: np.ndarray) -> dict:
    """
    fairface_gender_image_detection expects ViT-style preprocessing:
    - resize 224x224
    - rescale by 1/255
    - normalize mean=0.5 std=0.5 => (x - 0.5)/0.5
    - output index: 0=Female, 1=Male
    """
    if face_bgr_224.shape[:2] != (224, 224):
        face_bgr_224 = cv2.resize(face_bgr_224, (224, 224))

    rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - 0.5) / 0.5  # -> [-1, 1]
    inp = np.transpose(x, (2, 0, 1))[None, :, :, :].astype(np.float32)

    sess = _session2()
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: inp})[0]
    logits = np.asarray(out).reshape(-1)
    if logits.size < 2:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "model": "fairface_gender_onnx",
            "error": "Unexpected fairface model output shape",
        }

    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -40.0, 40.0))
    probs = (e / max(float(e.sum()), 1e-9)).astype(np.float64)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = _GENDER_LABELS2[idx] if 0 <= idx < len(_GENDER_LABELS2) else "unknown"

    return {
        "label": label,
        "confidence": round(conf, 3),
        "low_confidence": conf < 0.55,
        "model": "fairface_gender_image_detection_onnx",
        "prob_male": round(float(probs[1]), 4),
        "prob_female": round(float(probs[0]), 4),
    }


def _image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
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
            raise ValueError("Could not decode image for gender estimation")
        if bgr.ndim == 2:
            return cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        return bgr


def _face_roi_from_landmarks(
    bgr: np.ndarray,
    landmark_points: list[tuple[int, int]],
    pad_ratio: float = 0.18,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    if not landmark_points or len(landmark_points) < 468:
        raise ValueError("Insufficient landmarks for face crop")
    xs = [p[0] for p in landmark_points[:468]]
    ys = [p[1] for p in landmark_points[:468]]
    x1, x2 = max(min(xs) - 8, 0), min(max(xs) + 8, w - 1)
    y1, y2 = max(min(ys) - 8, 0), min(max(ys) + 8, h - 1)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w - 1)
    y2 = min(y2 + pad_y, h - 1)
    roi = bgr[y1 : y2 + 1, x1 : x2 + 1]
    if roi.size == 0:
        raise ValueError("Empty landmark crop")
    return roi


def _largest_face_roi_bgr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(48, 48))
    if len(faces) == 0:
        h, w = bgr.shape[:2]
        s = int(min(h, w) * 0.72)
        y0 = max((h - s) // 2, 0)
        x0 = max((w - s) // 2, 0)
        return bgr[y0 : y0 + s, x0 : x0 + s]
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    pad = int(0.12 * max(fw, fh))
    H, W = bgr.shape[:2]
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + fw + pad, W - 1), min(y + fh + pad, H - 1)
    return bgr[y1 : y2 + 1, x1 : x2 + 1]


def predict_gender_from_bytes(
    image_bytes: bytes,
    landmark_points: Optional[List[Tuple[int, int]]] = None,
) -> dict:
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

    H, W = bgr.shape[:2]
    try:
        if landmark_points and len(landmark_points) >= 468:
            face = _face_roi_from_landmarks(bgr, landmark_points)
        else:
            face = _largest_face_roi_bgr(bgr)
    except Exception:
        face = _largest_face_roi_bgr(bgr)

    face = cv2.resize(face, (224, 224))
    # Model zoo gender GoogLeNet: blob = NCHW BGR, scale 1, mean (104, 117, 123)
    blob = cv2.dnn.blobFromImage(
        face, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=False
    )

    sess = _session()
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: blob})[0]
    logits = np.asarray(out).reshape(-1)
    if logits.size < 2:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "low_confidence": True,
            "model": "googlenet_gender_adience_onnx",
            "error": "Unexpected model output shape",
        }

    z = logits - float(np.max(logits))
    e = np.exp(np.clip(z, -40.0, 40.0))
    probs = (e / max(float(e.sum()), 1e-9)).astype(np.float64)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = _GENDER_LABELS[idx] if 0 <= idx < len(_GENDER_LABELS) else "unknown"
    res = {
        "label": label,
        "confidence": round(conf, 3),
        "low_confidence": conf < 0.55,
        "model": "googlenet_gender_adience_onnx",
        "prob_male": round(float(probs[0]), 4) if probs.size > 1 else None,
        "prob_female": round(float(probs[1]), 4) if probs.size > 1 else None,
    }

    # If GoogLeNet is unsure, fall back to a FairFace-based gender classifier.
    if res["low_confidence"]:
        try:
            face_224 = cv2.resize(face, (224, 224))
            res2 = _predict_gender_fairface_onnx(face_224)
            # Prefer the fallback if it is more confident or the first was "unknown".
            if res2.get("label") != "unknown" and res2.get("confidence", 0.0) >= res.get("confidence", 0.0):
                return res2
        except Exception:
            pass

    return res
