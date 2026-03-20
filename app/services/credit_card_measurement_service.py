from __future__ import annotations

import io
import math
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

L_IRIS_IDX = (468, 469, 470, 471, 472)
R_IRIS_IDX = (473, 474, 475, 476, 477)

_LANDMARK_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "face_landmarker"
_LANDMARK_MODEL_FILE = _LANDMARK_MODEL_DIR / "face_landmarker.task"
_LANDMARK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
    "float16/1/face_landmarker.task"
)

_LANDMARK_LOCK = threading.Lock()
_FACE_LANDMARKER: FaceLandmarker | None = None


def _ensure_face_landmarker_model() -> None:
    _LANDMARK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _LANDMARK_MODEL_FILE.exists() and _LANDMARK_MODEL_FILE.stat().st_size > 1_000_000:
        return
    with urllib.request.urlopen(_LANDMARK_MODEL_URL, timeout=120) as r:  # noqa: S310
        _LANDMARK_MODEL_FILE.write_bytes(r.read())


def _get_face_landmarker() -> FaceLandmarker:
    global _FACE_LANDMARKER
    with _LANDMARK_LOCK:
        if _FACE_LANDMARKER is None:
            _ensure_face_landmarker_model()
            base_options = BaseOptions(model_asset_path=str(_LANDMARK_MODEL_FILE))
            options = FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            _FACE_LANDMARKER = FaceLandmarker.create_from_options(options)
        return _FACE_LANDMARKER


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"))
    except Exception:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _euclid_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _iris_center_and_diameter_px(
    points: list[tuple[int, int]], idxs: tuple[int, ...]
) -> tuple[tuple[float, float], float]:
    arr = np.asarray([[points[i][0], points[i][1]] for i in idxs], dtype=np.float32)
    cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    (_, _), r = cv2.minEnclosingCircle(arr)
    diam = max(2.0 * float(r), 1e-3)
    return (cx, cy), diam


class CreditCardMeasurementService:
    CARD_WIDTH_MM = 85.6  # ISO 7810

    @staticmethod
    def detect_credit_card_width_px(img_rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_width = None
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            area = w * h
            if 1.5 < aspect_ratio < 1.7 and area > 5000:
                width_px = max(w, h)
                if best_width is None or width_px > best_width:
                    best_width = width_px

        if best_width is None:
            raise ValueError("Credit card not detected. Ensure full card visibility.")

        return float(best_width)

    @staticmethod
    def classify_face_shape(width: float, height: float, chin_width: float) -> str:
        ratio = width / max(height, 1e-6)
        if ratio > 0.9:
            return "round"
        if 0.85 <= ratio <= 0.9:
            return "square"
        if 0.75 <= ratio < 0.85:
            return "oval"
        if chin_width < width * 0.7:
            return "heart"
        return "rectangle"

    @staticmethod
    def process(image_bytes: bytes) -> dict:
        rgb = _decode_rgb(image_bytes)
        h, w = rgb.shape[:2]

        card_width_px = CreditCardMeasurementService.detect_credit_card_width_px(rgb)
        mm_per_pixel = CreditCardMeasurementService.CARD_WIDTH_MM / card_width_px

        data = np.ascontiguousarray(rgb.astype(np.uint8))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
        result = _get_face_landmarker().detect(mp_image)
        if not result.face_landmarks:
            raise ValueError("No face detected")

        face = result.face_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face]

        if len(points) < 478:
            raise ValueError("Iris landmarks not available")

        (l_cx, l_cy), _ = _iris_center_and_diameter_px(points, L_IRIS_IDX)
        (r_cx, r_cy), _ = _iris_center_and_diameter_px(points, R_IRIS_IDX)
        pd_px = _euclid_px((l_cx, l_cy), (r_cx, r_cy))
        pd_mm = pd_px * mm_per_pixel

        mid_x = (points[133][0] + points[362][0]) / 2.0
        left_px = abs(l_cx - mid_x)
        right_px = abs(r_cx - mid_x)
        left_mm = left_px * mm_per_pixel
        right_mm = right_px * mm_per_pixel
        mono_sum = left_mm + right_mm
        if mono_sum > 1e-6:
            k = pd_mm / mono_sum
            left_mm *= k
            right_mm *= k

        jaw_left = points[234]
        jaw_right = points[454]
        chin = points[152]
        forehead = points[10]

        face_width_mm = _euclid_px(jaw_left, jaw_right) * mm_per_pixel
        face_height_mm = _euclid_px(chin, forehead) * mm_per_pixel
        chin_width_mm = _euclid_px(points[132], points[361]) * mm_per_pixel

        face_shape = CreditCardMeasurementService.classify_face_shape(
            face_width_mm, face_height_mm, chin_width_mm
        )

        return {
            "scale": {
                "reference": "credit_card",
                "card_width_px": round(card_width_px, 1),
                "mm_per_pixel": round(mm_per_pixel, 6),
                "pd_px_euclidean": round(pd_px, 3),
            },
            "mm": {
                "pd": round(pd_mm * 2.0) / 2.0,
                "pd_left": round(left_mm * 2.0) / 2.0,
                "pd_right": round(right_mm * 2.0) / 2.0,
                "nose_left": round(left_mm, 1),
                "nose_right": round(right_mm, 1),
                "face_width": round(face_width_mm, 1),
                "face_height": round(face_height_mm, 1),
                "face_ratio": round(face_width_mm / max(face_height_mm, 1e-6), 2),
                "nose_bridge_left": round(
                    _euclid_px(points[1], points[94]) * mm_per_pixel, 1
                ),
                "nose_bridge_right": round(
                    _euclid_px(points[1], points[331]) * mm_per_pixel, 1
                ),
            },
            "face_shape": face_shape,
        }
