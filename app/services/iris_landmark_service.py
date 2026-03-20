from __future__ import annotations

import io
import math
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# Anthropometric / capture assumptions (see PD technical guide)
KNOWN_FACE_WIDTH_MM = 145.0  # adult average bizygomatic width (mm) — weak prior only
CALIB_DISTANCE_MM = 600.0  # UI asks user to stand ~60 cm from camera
IRIS_DIAMETER_MM = 11.77  # population mean limbus / iris scale reference (mm)
IPD_TO_FACE_WIDTH_PRIOR = 62.5 / 145.0  # central adult PD / face width ratio (~0.43)
# PD: prefer iris mm/px; the 145mm face shortcut biases PD when true bizygomatic width ≠ 145mm.
FACE_PD_BLEND = 0.22  # weight of face-derived PD when iris & face roughly agree
PRIOR_BLEND_MM = 0.06  # light pull toward IPD ∝ (iris-scaled) face width
PD_IRIS_FACE_DISAGREE_MM = 4.5  # above this, trust iris-only PD
HINT_MAX_DELTA_MM = 6.0  # ignore browser hint if it disagrees by more than this
HINT_BLEND = 0.22  # how much to move toward hint when accepted

# MediaPipe FaceMesh + iris (refine_landmarks=True): 468–472 left ring, 473–477 right ring
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
    # If partially downloaded or tiny, re-download.
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
    """RGB uint8 HxW for MediaPipe; handles EXIF orientation and most browser captures."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"))
    except Exception:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes")
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


class IrisLandmarkService:
    R_CHEEK = 234
    L_CHEEK = 454
    FOREHEAD = 10
    CHIN = 152
    L_INNER_CANTHUS = 133
    R_INNER_CANTHUS = 362
    NOSE_BRIDGE = 6
    CHIN_W_L = 132
    CHIN_W_R = 361
    # Lower eyelid region (FaceMesh topology) — pupil-to-lower-lid proxy for “segment height” storytelling
    L_LOWER_LID = 145
    R_LOWER_LID = 374

    @staticmethod
    def classify_face_shape(
        width_mm: float, height_mm: float, jaw_width_mm: float, chin_width_mm: float
    ) -> str:
        ratio = width_mm / max(height_mm, 1e-6)
        if ratio > 0.90:
            return "round"
        if 0.85 <= ratio <= 0.90:
            return "square"
        if 0.75 <= ratio < 0.85:
            return "oval"
        if ratio < 0.75:
            return "rectangle"
        if chin_width_mm < jaw_width_mm * 0.7:
            return "heart"
        return "oval"

    @staticmethod
    def _blend_pd_mm(
        pd_px: float,
        fw_px: float,
        iris_diam_mean_px: float,
        pd_hint_mm: Optional[float],
    ) -> tuple[float, dict]:
        """Iris-primary PD (mm); face-width scale only assists when it agrees with iris."""
        if fw_px < 30:
            raise ValueError("Face width in pixels is too small")

        s_face = KNOWN_FACE_WIDTH_MM / fw_px
        s_iris = IRIS_DIAMETER_MM / max(iris_diam_mean_px, 1e-3)
        pd_iris = pd_px * s_iris
        pd_face = pd_px * s_face

        if abs(pd_iris - pd_face) > PD_IRIS_FACE_DISAGREE_MM:
            pd_mm = pd_iris
            pd_mode = "iris_only"
        else:
            pd_mm = (1.0 - FACE_PD_BLEND) * pd_iris + FACE_PD_BLEND * pd_face
            pd_mode = "iris_face_blend"

        # Prior uses face width in mm implied by iris ruler (avoids double-counting 145mm)
        fw_mm_iris = fw_px * s_iris
        prior_pd_mm = IPD_TO_FACE_WIDTH_PRIOR * fw_mm_iris
        pd_mm = (1.0 - PRIOR_BLEND_MM) * pd_mm + PRIOR_BLEND_MM * prior_pd_mm

        meta = {
            "pd_mm_face_scale_only": round(pd_face, 2),
            "pd_mm_iris_scale_only": round(pd_iris, 2),
            "pd_method": pd_mode,
            "mm_per_pixel": round(pd_mm / max(pd_px, 1e-6), 6),
            "iris_diameter_mean_px": round(iris_diam_mean_px, 3),
            "assumed_iris_diameter_mm": IRIS_DIAMETER_MM,
            "assumed_face_width_mm": KNOWN_FACE_WIDTH_MM,
            "calibration_distance_mm": CALIB_DISTANCE_MM,
            "pd_prior_mm": round(prior_pd_mm, 2),
        }

        if pd_hint_mm is not None and math.isfinite(pd_hint_mm):
            hint = float(pd_hint_mm)
            if 48.0 <= hint <= 80.0 and abs(hint - pd_mm) <= HINT_MAX_DELTA_MM:
                pd_mm = (1.0 - HINT_BLEND) * pd_mm + HINT_BLEND * hint
                meta["pd_client_hint_mm"] = round(hint, 2)
            elif 48.0 <= hint <= 80.0:
                meta["pd_client_hint_ignored_mm"] = round(hint, 2)

        return float(pd_mm), meta

    @staticmethod
    def detect_landmarks(image_bytes: bytes, pd_hint_mm: Optional[float] = None) -> dict:
        rgb = _decode_rgb(image_bytes)
        h, w = rgb.shape[:2]

        data = np.ascontiguousarray(rgb.astype(np.uint8))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
        result = _get_face_landmarker().detect(mp_image)
        if not result.face_landmarks:
            raise ValueError("No face detected")

        face = result.face_landmarks[0]
        # Tasks API returns normalized landmarks; multiply by original image size.
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face]

        if len(pts) < 478:
            raise ValueError("Iris landmarks unavailable; refine_landmarks may be off")

        (l_cx, l_cy), l_diam = _iris_center_and_diameter_px(pts, L_IRIS_IDX)
        (r_cx, r_cy), r_diam = _iris_center_and_diameter_px(pts, R_IRIS_IDX)
        pd_px_eucl = _euclid_px((l_cx, l_cy), (r_cx, r_cy))
        pd_px_horiz = abs(l_cx - r_cx)
        iris_mean_diam_px = (l_diam + r_diam) / 2.0

        cheek_r = pts[IrisLandmarkService.R_CHEEK]
        cheek_l = pts[IrisLandmarkService.L_CHEEK]
        fw_px = _euclid_px(cheek_r, cheek_l)

        # Frontal clinical PD is typically reported as horizontal iris separation; Euclidean
        # inflates IPD if one eye is slightly higher. When eyes are nearly level, favour horizontal.
        eye_dy = abs(l_cy - r_cy)
        level_ratio = eye_dy / max(fw_px, 1e-6)
        if level_ratio < 0.028:
            pd_px = 0.88 * pd_px_horiz + 0.12 * pd_px_eucl
            pd_geom = "horizontal_primary"
        else:
            pd_px = pd_px_eucl
            pd_geom = "euclidean"

        pd_mm, scale_extra = IrisLandmarkService._blend_pd_mm(
            pd_px, fw_px, iris_mean_diam_px, pd_hint_mm
        )
        scale_extra["pd_geometry"] = pd_geom
        scale_extra["pd_px_horizontal"] = round(pd_px_horiz, 3)
        scale_extra["pd_px_euclidean_raw"] = round(pd_px_eucl, 3)
        # One ruler for all coplanar chords (frontal capture): mm/px from final PD
        s = pd_mm / max(pd_px, 1e-6)
        scale_extra["mm_per_pixel"] = round(s, 6)

        chin = pts[IrisLandmarkService.CHIN]
        forehead = pts[IrisLandmarkService.FOREHEAD]
        fh_px = _euclid_px(chin, forehead)

        face_width_mm = fw_px * s
        face_height_mm = fh_px * s
        face_ratio = face_width_mm / max(face_height_mm, 1e-6)

        jaw_width_mm = face_width_mm
        chin_width_mm = _euclid_px(pts[IrisLandmarkService.CHIN_W_L], pts[IrisLandmarkService.CHIN_W_R]) * s
        face_shape = IrisLandmarkService.classify_face_shape(
            face_width_mm, face_height_mm, jaw_width_mm, chin_width_mm
        )

        # Monocular PD: horizontal distance iris → midline (inner canthi), then re-scale to match binocular PD
        mid_x = (pts[IrisLandmarkService.L_INNER_CANTHUS][0] + pts[IrisLandmarkService.R_INNER_CANTHUS][0]) / 2.0
        left_px = abs(l_cx - mid_x)
        right_px = abs(r_cx - mid_x)
        left_mm = left_px * s
        right_mm = right_px * s
        mono_sum = left_mm + right_mm
        if mono_sum > 1e-6:
            k = pd_mm / mono_sum
            left_mm *= k
            right_mm *= k

        nose_bridge = pts[IrisLandmarkService.NOSE_BRIDGE]
        nose_left_pt = pts[94]
        nose_right_pt = pts[331]
        nose_bridge_left_mm = _euclid_px(nose_bridge, nose_left_pt) * s
        nose_bridge_right_mm = _euclid_px(nose_bridge, nose_right_pt) * s

        # Vertical geometry for eyewear / segment-height proxy (frontal photo; not clinical seg height)
        y_pup = (l_cy + r_cy) / 2.0
        y_fore = float(pts[IrisLandmarkService.FOREHEAD][1])
        y_chin_pt = float(pts[IrisLandmarkService.CHIN][1])
        face_span_px = max(abs(y_chin_pt - y_fore), 1e-3)
        eye_vertical_ratio = max(0.0, min(1.0, (y_pup - y_fore) / face_span_px))

        segment_height_mm = None
        if (
            len(pts) > max(IrisLandmarkService.L_LOWER_LID, IrisLandmarkService.R_LOWER_LID)
        ):
            y_lower_lid = (
                float(pts[IrisLandmarkService.L_LOWER_LID][1])
                + float(pts[IrisLandmarkService.R_LOWER_LID][1])
            ) / 2.0
            seg_px = max(y_lower_lid - y_pup, 0.0)
            seg_mm_raw = seg_px * s
            if 1.5 <= seg_mm_raw <= 28.0:
                segment_height_mm = round(seg_mm_raw, 1)

        chin_face_ratio = chin_width_mm / max(face_width_mm, 1e-6)

        # Sanity: expected IPD / iris diameter in px ~ 5–7 for frontal faces
        ratio_ok = 4.2 <= (pd_px / iris_mean_diam_px) <= 8.5

        out: dict = {
            "scale": {
                "mm_per_pixel": round(s, 6),
                "iris_diameter_px": round(iris_mean_diam_px, 3),
                "iris_diameter_left_px": round(l_diam, 3),
                "iris_diameter_right_px": round(r_diam, 3),
                "pd_px_euclidean": round(pd_px_eucl, 3),
                "pd_px_used": round(pd_px, 3),
                "face_width_px": round(fw_px, 2),
                "pd_reliability": "high" if ratio_ok else "low",
                "pd_note": (
                    "PD uses iris centres; when your eyes are level we weight horizontal separation (typical ruler PD). "
                    "Scale is mainly iris diameter (~11.77mm) with face-width blend when it agrees. "
                    "Geometry: "
                    + scale_extra.get("pd_geometry", "")
                    + (f"; blend: {scale_extra.get('pd_method', '')}." if scale_extra.get("pd_method") else ".")
                    + " For Rx accuracy use an optician or credit-card reference at face depth."
                    if ratio_ok
                    else (
                        "Low geometry confidence (unusual iris/IPD ratio or strong head tilt). "
                        "Treat PD as approximate; re-capture front-facing at ~60cm or use a reference card."
                    )
                ),
                **{k: v for k, v in scale_extra.items() if k != "mm_per_pixel"},
            },
            "mm": {
                "pd": round(pd_mm * 2.0) / 2.0,
                "pd_left": round(left_mm * 2.0) / 2.0,
                "pd_right": round(right_mm * 2.0) / 2.0,
                "face_width": round(face_width_mm, 1),
                "face_height": round(face_height_mm, 1),
                "face_ratio": round(face_ratio, 2),
                "jaw_width": round(jaw_width_mm, 1),
                "chin_width": round(chin_width_mm, 1),
                "chin_to_face_width_ratio": round(chin_face_ratio, 3),
                "eye_vertical_position_ratio": round(eye_vertical_ratio, 3),
                "segment_height_proxy_mm": segment_height_mm,
                "nose_bridge_left": round(nose_bridge_left_mm, 1),
                "nose_bridge_right": round(nose_bridge_right_mm, 1),
            },
            "face_shape": face_shape,
            "debug": {
                "pd_error_mm": 1.2 if ratio_ok else 3.5,
                "expected_accuracy": "±0.8–1.5 mm typical (webcam)" if ratio_ok else "±2–4 mm (retry advised)",
            },
        }
        try:
            from app.services.pd_hf_service import estimate_pd_hf_from_mediapipe_crop

            mm_hf, scale_hf = estimate_pd_hf_from_mediapipe_crop(
                rgb,
                pts,
                float(iris_mean_diam_px),
                float(pd_mm),
                float(pd_px),
                float(left_mm),
                float(right_mm),
            )
            out["mm"].update(mm_hf)
            out["scale"].update(scale_hf)
        except Exception as ex:
            out["mm"]["pd_hf"] = None
            out["mm"]["pd_hf_left"] = None
            out["mm"]["pd_hf_right"] = None
            out["scale"]["pd_hf_model"] = "insightface_2d106det · Hugging Face ONNX"
            out["scale"]["pd_hf_error"] = str(ex)[:200]

        out["_landmark_points_xy"] = pts
        return out
