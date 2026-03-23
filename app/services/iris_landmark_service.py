from __future__ import annotations

import io
import json
import math
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Optional

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
PD_IRIS_FACE_DISAGREE_MM = 4.5  # above this, trust iris-only PD (unless iris looks broken — see _blend_pd_mm)
HINT_MAX_DELTA_MM = 6.0  # ignore browser hint if it disagrees by more than this
HINT_BLEND = 0.22  # how much to move toward hint when accepted
# Frontal webcam: IPD_px / iris_diameter_px is usually ~5–7. Above ~6.65, iris Ø is often underestimated
# (tight limbus model) → mm/px too large → PD reads high (e.g. 87 mm vs true ~60).
IPD_OVER_IRIS_DIAM_RATIO_WARN = 6.65
IPD_OVER_IRIS_DIAM_RATIO_TARGET = 5.12  # geometric mean-ish; used to inflate iris Ø toward plausible ratio
IPD_IRIS_DIAM_MAX_SCALEUP = 1.52  # cap correction factor

# Typical adult binocular PD band (optical retail / clinical screening) — for AR UX + confidence, not forced clamping
PD_ADULT_MIN_MM = 54.0
PD_ADULT_MAX_MM = 74.0
# Pediatric / small-head (webcam heuristic — not clinical age): child PD is usually ~40–58 mm, not 54–74 mm.
PEDiatric_FACE_MM_IRIS_MAX = 118.0  # iris-scaled cheek width below this → likely child / small teen
PEDiatric_IPD_OVER_FACE_MAX = 0.37  # IPD/fw below this often indicates large eyes vs cheek width (common in kids)
# Cheek landmarks (234/454) often read wide vs true bizygomatic width; iris-scale face mm can look "adult"
# while IPD/cheek span stays relatively high — common for young faces at arm's length.
PEDiatric_FACE_MM_IRIS_WIDE_MAX = 172.0
PEDiatric_IPD_TO_CHEEK_MIN = 0.46  # pd_px / cheek span; nominal adult ~0.43; elevated suggests this pattern
IRIS_DIAMETER_MM_PEDIATRIC = 11.12  # child limbus ~same as adult but model error skews high PD if we use 11.77
PEDiatric_IPD_TO_FACE_RATIO = 0.415  # ~PD/face for young children (below adult 62.5/145)
PEDiatric_PRIOR_BLEND = 0.48  # blend toward pediatric prior when heuristic fires

# Typical child binocular PD band (very rough screening)
PD_PEDIATRIC_MIN_MM = 40.0
PD_PEDIATRIC_MAX_MM = 58.0

# MediaPipe FaceMesh + iris (refine_landmarks=True): 468–472 left ring, 473–477 right ring
# Order: center, top, bottom, left, right (cardinal edges — NOT eye corners 33/133/263/362)
L_IRIS_IDX = (468, 469, 470, 471, 472)
R_IRIS_IDX = (473, 474, 475, 476, 477)
# Eye aperture (horizontal) for sanity: iris diameter must be a fraction of this, not the whole eye
L_EYE_OUTER_INNER = (33, 133)   # left canthi
R_EYE_OUTER_INNER = (263, 362)  # right canthi
# Typical iris horizontal diameter / eye opening width for frontal view (~0.35–0.50)
IRIS_DIAM_MIN_FRAC_EYE = 0.20
IRIS_DIAM_MAX_FRAC_EYE = 0.52


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


def _pd_in_typical_adult_range_mm(pd_mm: float) -> bool:
    return PD_ADULT_MIN_MM <= pd_mm <= PD_ADULT_MAX_MM


def _ar_pd_geometry_quality(
    ratio_ok: bool,
    iris_left_sanity: str,
    iris_right_sanity: str,
    pd_mm: float,
    *,
    likely_pediatric: bool = False,
) -> str:
    """Web-AR style quality tier: alignment + iris scale validity + adult or pediatric PD band."""
    if likely_pediatric:
        in_band = PD_PEDIATRIC_MIN_MM <= pd_mm <= PD_PEDIATRIC_MAX_MM
    else:
        in_band = _pd_in_typical_adult_range_mm(pd_mm)
    iris_ok = iris_left_sanity == "ok" and iris_right_sanity == "ok"
    if ratio_ok and iris_ok and in_band:
        return "excellent"
    if ratio_ok and in_band:
        return "good"
    if ratio_ok:
        return "fair"
    return "fair"


def _trace_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _build_pd_calculation_trace(
    *,
    image_w: int,
    image_h: int,
    left_iris_center_px: tuple[float, float],
    right_iris_center_px: tuple[float, float],
    iris_diameter_left_px: float,
    iris_diameter_right_px: float,
    iris_diameter_mean_px: float,
    pd_px_horizontal: float,
    pd_px_euclidean: float,
    eye_dy_px: float,
    face_width_cheek_px: float,
    level_ratio: float,
    pd_geometry: str,
    pd_px_used: float,
    pd_hint_mm: Optional[float],
    scale_extra: dict,
    pd_mm_before_round: float,
    pd_mm_rounded_half_mm: float,
    ratio_ok: bool,
    extra_scale_keys: Optional[dict] = None,
) -> dict[str, Any]:
    """End-to-end PD math for API `debug.pd_calculation_trace` + optional server stdout."""
    s_face = KNOWN_FACE_WIDTH_MM / max(face_width_cheek_px, 1e-6)
    s_iris = IRIS_DIAMETER_MM / max(iris_diameter_mean_px, 1e-3)
    pd_iris = pd_px_used * s_iris
    pd_face = pd_px_used * s_face
    disagree = abs(pd_iris - pd_face)
    fw_mm_iris = face_width_cheek_px * s_iris
    prior_pd = IPD_TO_FACE_WIDTH_PRIOR * fw_mm_iris

    formulas = [
        "STEP A — Image & landmarks (MediaPipe Face Landmarker, full-res px):",
        f"  image_size_px = {image_w} × {image_h}",
        "STEP B — Iris ring → centre (mean of 5 pts) & diameter (horizontal/vertical iris edges; not min circle on whole eye):",
        f"  left_iris_center_px  = ({left_iris_center_px[0]:.2f}, {left_iris_center_px[1]:.2f})",
        f"  right_iris_center_px = ({right_iris_center_px[0]:.2f}, {right_iris_center_px[1]:.2f})",
        f"  iris_diameter_left_px  = {iris_diameter_left_px:.3f}",
        f"  iris_diameter_right_px = {iris_diameter_right_px:.3f}",
        f"  iris_diameter_mean_px = (L+R)/2 = {iris_diameter_mean_px:.3f}",
        "STEP C — IPD chord in pixels (same image coordinates):",
        f"  pd_px_horizontal = |left_cx - right_cx| = {pd_px_horizontal:.3f}",
        f"  pd_px_euclidean  = hypot(Δx,Δy) between iris centres = {pd_px_euclidean:.3f}",
        f"  eye_vertical_delta_px = |left_cy - right_cy| = {eye_dy_px:.3f}",
        f"  face_width_cheek_px = euclidean(cheek 234, cheek 454) = {face_width_cheek_px:.3f}",
        f"  level_ratio = eye_dy / face_width_cheek = {level_ratio:.5f} (threshold 0.028)",
    ]
    if pd_geometry == "horizontal_primary":
        formulas.append(
            "  pd_px_used = 0.88 × pd_px_horizontal + 0.12 × pd_px_euclidean  "
            f"(eyes level) = {pd_px_used:.3f}"
        )
    else:
        formulas.append(f"  pd_px_used = pd_px_euclidean (tilted head) = {pd_px_used:.3f}")

    formulas.extend(
        [
            "STEP D — Two mm/px rulers (anthropometric priors):",
            f"  s_iris = IRIS_DIAMETER_MM / iris_diameter_mean_px = {IRIS_DIAMETER_MM} / {iris_diameter_mean_px:.3f} = {s_iris:.6f} mm/px",
            f"  s_face = KNOWN_FACE_WIDTH_MM / face_width_cheek_px = {KNOWN_FACE_WIDTH_MM} / {face_width_cheek_px:.2f} = {s_face:.6f} mm/px",
            "STEP E — PD in mm from each ruler:",
            f"  pd_iris_mm = pd_px_used × s_iris = {pd_iris:.3f}",
            f"  pd_face_mm = pd_px_used × s_face = {pd_face:.3f}",
            f"  |pd_iris - pd_face| = {disagree:.3f} mm (disagree threshold {PD_IRIS_FACE_DISAGREE_MM} mm)",
        ]
    )

    mode = scale_extra.get("pd_method", "?")
    if disagree > PD_IRIS_FACE_DISAGREE_MM:
        formulas.append(f"  → pd_mm = pd_iris only (iris_only), blend weight face = 0")
    else:
        formulas.append(
            f"  → pd_mm = (1-{FACE_PD_BLEND})×pd_iris + {FACE_PD_BLEND}×pd_face "
            f"(iris_face_blend)"
        )

    formulas.extend(
        [
            "STEP F — Face-width prior on IPD (light):",
            f"  face_width_mm_iris_ruler = fw_px × s_iris = {fw_mm_iris:.2f}",
            f"  prior_pd_mm = IPD_TO_FACE_WIDTH_PRIOR × face_width_mm_iris_ruler "
            f"({IPD_TO_FACE_WIDTH_PRIOR:.5f} ≈ 62.5/145) = {prior_pd:.2f}",
            f"  pd_mm = (1-{PRIOR_BLEND_MM})×pd_after_step_E + {PRIOR_BLEND_MM}×prior_pd_mm",
        ]
    )

    hint_applied = scale_extra.get("pd_client_hint_mm")
    hint_ignored = scale_extra.get("pd_client_hint_ignored_mm")
    if hint_applied is not None:
        formulas.append(
            f"STEP G — Browser pd_hint_mm = {hint_applied}: blended in at weight {HINT_BLEND} "
            f"(within ±{HINT_MAX_DELTA_MM} mm of server PD)."
        )
    elif hint_ignored is not None:
        formulas.append(
            f"STEP G — Browser pd_hint_mm = {hint_ignored} ignored (outside ±{HINT_MAX_DELTA_MM} mm or range)."
        )
    elif pd_hint_mm is not None and math.isfinite(float(pd_hint_mm)):
        formulas.append(f"STEP G — pd_hint_mm was present but not applied (see scale metadata).")
    else:
        formulas.append("STEP G — No client pd_hint_mm for this request.")

    formulas.extend(
        [
            f"STEP H — Final continuous PD mm ≈ {_trace_float(pd_mm_before_round):.4f}",
            f"STEP I — Display PD = round(pd_mm×2)/2 to 0.5 mm → {pd_mm_rounded_half_mm}",
            f"STEP J — Global mm/px for face chords: s = pd_mm / pd_px_used = {_trace_float(pd_mm_before_round) / max(pd_px_used, 1e-9):.6f}",
            "STEP K — Sanity: IPD_px / iris_diam_px ratio should be ~5–7 frontal:",
            f"  ratio = {pd_px_used / max(iris_diameter_mean_px, 1e-6):.3f} → reliability {'high' if ratio_ok else 'low'}",
        ]
    )

    trace: dict[str, Any] = {
        "summary": "Primary binocular PD from iris centres; scale = iris diameter (11.77 mm) with optional face-width blend and light prior; optional browser hint.",
        "constants": {
            "IRIS_DIAMETER_MM": IRIS_DIAMETER_MM,
            "KNOWN_FACE_WIDTH_MM": KNOWN_FACE_WIDTH_MM,
            "IPD_TO_FACE_WIDTH_PRIOR": round(IPD_TO_FACE_WIDTH_PRIOR, 6),
            "FACE_PD_BLEND": FACE_PD_BLEND,
            "PRIOR_BLEND_MM": PRIOR_BLEND_MM,
            "PD_IRIS_FACE_DISAGREE_MM": PD_IRIS_FACE_DISAGREE_MM,
            "HINT_BLEND": HINT_BLEND,
            "HINT_MAX_DELTA_MM": HINT_MAX_DELTA_MM,
            "CALIB_DISTANCE_MM_UI_hint": CALIB_DISTANCE_MM,
            "PD_ADULT_MIN_MM": PD_ADULT_MIN_MM,
            "PD_ADULT_MAX_MM": PD_ADULT_MAX_MM,
        },
        "pixels": {
            "image_width": image_w,
            "image_height": image_h,
            "left_iris_center": [round(left_iris_center_px[0], 2), round(left_iris_center_px[1], 2)],
            "right_iris_center": [round(right_iris_center_px[0], 2), round(right_iris_center_px[1], 2)],
            "iris_diameter_left": round(iris_diameter_left_px, 3),
            "iris_diameter_right": round(iris_diameter_right_px, 3),
            "iris_diameter_mean": round(iris_diameter_mean_px, 3),
            "pd_px_horizontal": round(pd_px_horizontal, 3),
            "pd_px_euclidean": round(pd_px_euclidean, 3),
            "pd_px_used": round(pd_px_used, 3),
            "face_width_cheek_px": round(face_width_cheek_px, 2),
            "eye_vertical_delta_px": round(eye_dy_px, 3),
            "level_ratio": round(level_ratio, 6),
            "pd_geometry": pd_geometry,
        },
        "intermediate_mm": {
            "s_iris_mm_per_px": round(s_iris, 6),
            "s_face_mm_per_px": round(s_face, 6),
            "pd_iris_mm": round(pd_iris, 3),
            "pd_face_mm": round(pd_face, 3),
            "pd_method": mode,
            "prior_pd_mm": round(prior_pd, 3),
            "pd_mm_before_round": round(pd_mm_before_round, 4),
            "pd_mm_display_half_step": pd_mm_rounded_half_mm,
            "ipd_px_over_iris_diam_px": round(
                pd_px_used / max(iris_diameter_mean_px, 1e-6), 3
            ),
        },
        "scale_extra_echo": {k: v for k, v in scale_extra.items() if k != "mm_per_pixel"},
        "formulas_plaintext": formulas,
    }

    if extra_scale_keys:
        trace["hf_and_extra_scale"] = {
            k: v for k, v in extra_scale_keys.items()
            if k.startswith("pd_hf") or k in ("pd_hf_model", "pd_hf_method", "pd_hf_note")
        }

    return trace


def _maybe_stdout_pd_trace(trace: dict[str, Any]) -> None:
    if os.environ.get("PD_TRACE_PRINT", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        print(
            "\n========== PD_CALCULATION_TRACE ==========\n",
            json.dumps(trace, indent=2, default=str),
            "\n==========================================\n",
            flush=True,
        )
    except Exception:
        pass


def _iris_center_and_diameter_px(
    points: list[tuple[int, int]], idxs: tuple[int, ...]
) -> tuple[tuple[float, float], float]:
    """Iris centre + diameter in px.

    Do **not** use minEnclosingCircle on all five points: one bad edge or confused
    landmarks can inflate the circle to the whole visible eye. We use horizontal
    and vertical chords between iris edge landmarks (469–472 / 474–477), which
    track the limbus ring MediaPipe intended for refine_landmarks iris mode.
    """
    c, t, b, le, ri = idxs
    arr = np.asarray([[points[i][0], points[i][1]] for i in idxs], dtype=np.float32)
    cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())

    pt = np.asarray([points[t][0], points[t][1]], dtype=np.float32)
    pb = np.asarray([points[b][0], points[b][1]], dtype=np.float32)
    pl = np.asarray([points[le][0], points[le][1]], dtype=np.float32)
    pr = np.asarray([points[ri][0], points[ri][1]], dtype=np.float32)

    d_h = float(np.linalg.norm(pr - pl))
    d_v = float(np.linalg.norm(pb - pt))
    # Horizontal chord is more stable for PD scale; vertical is affected by gaze.
    diam_edges = 0.65 * d_h + 0.35 * d_v

    # Backup: smallest circle through the four edge points only (excludes centre).
    edge_only = np.asarray([pt, pb, pl, pr], dtype=np.float32)
    (_, _), r_enc = cv2.minEnclosingCircle(edge_only)
    diam_circle_edges = max(2.0 * float(r_enc), 1e-3)

    # Prefer edge chords; if circle-on-edges is wildly larger, trust chords.
    if diam_circle_edges > 1.25 * max(diam_edges, 1e-3):
        diam = diam_edges
    else:
        diam = float(0.5 * (diam_edges + diam_circle_edges))

    return (cx, cy), max(diam, 1e-3)


def _eye_opening_px(points: list[tuple[int, int]], outer_inner: tuple[int, int]) -> float:
    a, b = outer_inner
    return _euclid_px(points[a], points[b])


def _likely_pediatric(fw_mm_from_iris: float, pd_px: float, fw_px: float) -> bool:
    """Heuristic: small iris-scaled face width or child-like IPD vs cheek span — not a clinical age estimate."""
    if fw_mm_from_iris < PEDiatric_FACE_MM_IRIS_MAX:
        return True
    r = pd_px / max(fw_px, 1e-6)
    if fw_mm_from_iris < 128.0 and r < PEDiatric_IPD_OVER_FACE_MAX:
        return True
    # Inflated iris-scaled face width but high IPD/cheek — do not treat as typical adult head.
    if (
        PEDiatric_FACE_MM_IRIS_MAX <= fw_mm_from_iris < PEDiatric_FACE_MM_IRIS_WIDE_MAX
        and r >= PEDiatric_IPD_TO_CHEEK_MIN
    ):
        return True
    return False


def _correct_iris_mean_px_for_ipd_ratio(
    iris_mean_px: float,
    pd_px: float,
) -> tuple[float, str]:
    """When IPD/iris ratio is too high, iris diameter in px is usually underestimated — inflate slightly."""
    if pd_px <= 0 or iris_mean_px <= 0:
        return max(iris_mean_px, 1e-3), "skip"
    r = pd_px / iris_mean_px
    if r <= IPD_OVER_IRIS_DIAM_RATIO_WARN:
        return iris_mean_px, "ok"
    scale = min(r / max(IPD_OVER_IRIS_DIAM_RATIO_TARGET, 1e-6), IPD_IRIS_DIAM_MAX_SCALEUP)
    return max(iris_mean_px * scale, 1e-3), "ipd_ratio_iris_inflate"


def _adjust_iris_diameter_vs_eye(
    diam_px: float,
    eye_opening_px: float,
) -> tuple[float, str]:
    """If iris diameter looks like whole-eye width, clamp to plausible limbus range."""
    if eye_opening_px < 1e-3 or not math.isfinite(eye_opening_px):
        return max(diam_px, 1e-3), "no_eye_width"
    frac = diam_px / eye_opening_px
    if IRIS_DIAM_MIN_FRAC_EYE <= frac <= IRIS_DIAM_MAX_FRAC_EYE:
        return diam_px, "ok"
    if frac > IRIS_DIAM_MAX_FRAC_EYE:
        # Landmark ring likely matched eyelid / whole aperture — scale down.
        adj = eye_opening_px * ((IRIS_DIAM_MIN_FRAC_EYE + IRIS_DIAM_MAX_FRAC_EYE) / 2.0)
        return max(adj, 1e-3), "clamped_large_vs_eye"
    # Too-small reading: do NOT use IRIS_DIAM_MIN_FRAC_EYE (e.g. 0.2) — that is *below* a real
    # limbus/eye ratio (~0.35–0.45) and shrinks iris_px → inflates mm/px → PD reads ~1.4–1.6× high.
    adj = eye_opening_px * ((IRIS_DIAM_MIN_FRAC_EYE + IRIS_DIAM_MAX_FRAC_EYE) / 2.0)
    return max(adj, 1e-3), "clamped_small_vs_eye"


def _hough_iris_diameter_px(
    rgb: np.ndarray,
    center_xy: tuple[float, float],
    iris_lr_span_px: float,
) -> Optional[float]:
    """Optional pixel-intensity cross-check: circle fit inside eye ROI (OpenCV Hough)."""
    h, w = rgb.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    approx_r = max(int(iris_lr_span_px / 2), 6)
    pad = max(approx_r * 3, 32)
    x1 = max(cx - pad, 0)
    x2 = min(cx + pad, w)
    y1 = max(cy - pad, 0)
    y2 = min(cy + pad, h)
    roi = rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    rh, rw = gray.shape
    min_r = max(int(rh * 0.12), 5)
    max_r = min(int(rh * 0.50), int(rw * 0.50), approx_r + 25)
    if max_r <= min_r:
        max_r = min_r + 1

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(rh, rw),
        param1=60,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    best_r = None
    best_d = float("inf")
    cx_roi = cx - x1
    cy_roi = cy - y1
    for (hx, hy, r) in circles:
        d = math.hypot(hx - cx_roi, hy - cy_roi)
        if d < best_d and min_r <= r <= max_r:
            best_d = d
            best_r = r
    if best_r is None:
        return None
    # Reject if Hough centre drifted too far from MediaPipe centre
    if best_d > approx_r * 2.2:
        return None
    return float(2 * best_r)


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
        *,
        pediatric: bool = False,
    ) -> tuple[float, dict]:
        """Iris-primary PD (mm); face-width scale only assists when it agrees with iris."""
        if fw_px < 30:
            raise ValueError("Face width in pixels is too small")

        s_face = KNOWN_FACE_WIDTH_MM / fw_px
        iris_mm_ref = IRIS_DIAMETER_MM_PEDIATRIC if pediatric else IRIS_DIAMETER_MM
        s_iris = iris_mm_ref / max(iris_diam_mean_px, 1e-3)
        pd_iris = pd_px * s_iris
        pd_face = pd_px * s_face

        if abs(pd_iris - pd_face) > PD_IRIS_FACE_DISAGREE_MM:
            # Iris-only was meant for when 145mm face prior is wrong — but underestimated iris Ø also
            # drives pd_iris sky-high. If iris says "very high PD" and face says lower, blend toward face.
            if (
                pd_iris > PD_ADULT_MAX_MM
                and pd_face < pd_iris
                and pd_face >= 48.0
                and pd_iris - pd_face > 8.0
            ):
                w = 0.40
                pd_mm = w * pd_iris + (1.0 - w) * pd_face
                pd_mode = "iris_face_blend_iris_suspect"
            else:
                pd_mm = pd_iris
                pd_mode = "iris_only"
        else:
            pd_mm = (1.0 - FACE_PD_BLEND) * pd_iris + FACE_PD_BLEND * pd_face
            pd_mode = "iris_face_blend"

        # Prior uses face width in mm implied by iris ruler (avoids double-counting 145mm)
        fw_mm_iris = fw_px * s_iris
        prior_pd_mm = IPD_TO_FACE_WIDTH_PRIOR * fw_mm_iris
        pd_mm = (1.0 - PRIOR_BLEND_MM) * pd_mm + PRIOR_BLEND_MM * prior_pd_mm

        pd_pediatric_prior_mm: Optional[float] = None
        if pediatric:
            # Pull away from adult-only priors: child PD is much smaller than adult 54–74 mm band.
            pd_pediatric_prior_mm = PEDiatric_IPD_TO_FACE_RATIO * fw_mm_iris
            pd_mm = (1.0 - PEDiatric_PRIOR_BLEND) * pd_mm + PEDiatric_PRIOR_BLEND * pd_pediatric_prior_mm

        meta = {
            "pd_mm_face_scale_only": round(pd_face, 2),
            "pd_mm_iris_scale_only": round(pd_iris, 2),
            "pd_method": pd_mode,
            "mm_per_pixel": round(pd_mm / max(pd_px, 1e-6), 6),
            "iris_diameter_mean_px": round(iris_diam_mean_px, 3),
            "assumed_iris_diameter_mm": iris_mm_ref,
            "likely_pediatric_heuristic": pediatric,
            "assumed_face_width_mm": KNOWN_FACE_WIDTH_MM,
            "calibration_distance_mm": CALIB_DISTANCE_MM,
            "pd_prior_mm": round(prior_pd_mm, 2),
        }
        if pediatric and pd_pediatric_prior_mm is not None:
            meta["pd_pediatric_prior_mm"] = round(pd_pediatric_prior_mm, 2)

        if pd_hint_mm is not None and math.isfinite(pd_hint_mm):
            hint = float(pd_hint_mm)
            # Do not pull toward a client hint that matches a broken iris scale (both ~88–90 mm).
            hint_toxic = pd_iris > 82.0 and abs(hint - pd_iris) < 12.0
            if (
                48.0 <= hint <= 80.0
                and abs(hint - pd_mm) <= HINT_MAX_DELTA_MM
                and not hint_toxic
            ):
                pd_mm = (1.0 - HINT_BLEND) * pd_mm + HINT_BLEND * hint
                meta["pd_client_hint_mm"] = round(hint, 2)
            elif 48.0 <= hint <= 80.0:
                meta["pd_client_hint_ignored_mm"] = round(hint, 2)
                if hint_toxic:
                    meta["pd_client_hint_ignored_reason"] = "matches_inflated_iris_preview"

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

        eye_l_px = _eye_opening_px(pts, L_EYE_OUTER_INNER)
        eye_r_px = _eye_opening_px(pts, R_EYE_OUTER_INNER)
        l_diam, l_iris_sanity = _adjust_iris_diameter_vs_eye(l_diam, eye_l_px)
        r_diam, r_iris_sanity = _adjust_iris_diameter_vs_eye(r_diam, eye_r_px)

        # Optional intensity-based iris ring (validates landmarks are not spanning whole eye)
        l_lr_span = _euclid_px(pts[L_IRIS_IDX[3]], pts[L_IRIS_IDX[4]])
        r_lr_span = _euclid_px(pts[R_IRIS_IDX[3]], pts[R_IRIS_IDX[4]])
        hough_env = os.environ.get("IRIS_HOUGH", "").strip().lower() in ("1", "true", "yes")
        if hough_env or l_iris_sanity != "ok" or r_iris_sanity != "ok":
            h_l = _hough_iris_diameter_px(rgb, (l_cx, l_cy), l_lr_span)
            h_r = _hough_iris_diameter_px(rgb, (r_cx, r_cy), r_lr_span)
            if h_l is not None and l_iris_sanity != "ok":
                l_diam = float(0.45 * l_diam + 0.55 * h_l)
            elif h_l is not None and hough_env:
                l_diam = float(0.5 * (l_diam + h_l))
            if h_r is not None and r_iris_sanity != "ok":
                r_diam = float(0.45 * r_diam + 0.55 * h_r)
            elif h_r is not None and hough_env:
                r_diam = float(0.5 * (r_diam + h_r))

        # Binocular IPD in px: use iris *center* landmarks 468/473 (clinical), not the 5-point ring mean.
        l_ic = pts[468]
        r_ic = pts[473]
        pd_px_eucl = _euclid_px(l_ic, r_ic)
        pd_px_horiz = abs(l_ic[0] - r_ic[0])
        iris_raw_mean_px = (l_diam + r_diam) / 2.0

        cheek_r = pts[IrisLandmarkService.R_CHEEK]
        cheek_l = pts[IrisLandmarkService.L_CHEEK]
        fw_px = _euclid_px(cheek_r, cheek_l)

        # Frontal clinical PD is typically reported as horizontal iris separation; Euclidean
        # inflates IPD if one eye is slightly higher. When eyes are nearly level, favour horizontal.
        eye_dy = abs(l_ic[1] - r_ic[1])
        level_ratio = eye_dy / max(fw_px, 1e-6)
        if level_ratio < 0.028:
            pd_px = 0.88 * pd_px_horiz + 0.12 * pd_px_eucl
            pd_geom = "horizontal_primary"
        else:
            pd_px = pd_px_eucl
            pd_geom = "euclidean"

        # Iris IPD/Ø ratio fix: always apply for mm/px when iris Ø in px is underestimated (tight limbus).
        # Pediatric mode only changes priors / UX — skipping this step inflated PD for children (~80+ mm).
        fw_mm_est_iris = fw_px * IRIS_DIAMETER_MM / max(iris_raw_mean_px, 1e-3)
        likely_pediatric = _likely_pediatric(fw_mm_est_iris, pd_px, fw_px)
        iris_mean_diam_px, iris_ratio_note = _correct_iris_mean_px_for_ipd_ratio(
            iris_raw_mean_px, pd_px
        )

        pd_mm, scale_extra = IrisLandmarkService._blend_pd_mm(
            pd_px, fw_px, iris_mean_diam_px, pd_hint_mm, pediatric=likely_pediatric
        )
        scale_extra["pd_pediatric_band_mm"] = f"{PD_PEDIATRIC_MIN_MM:g}–{PD_PEDIATRIC_MAX_MM:g}"
        scale_extra["pd_in_typical_pediatric_range"] = (
            PD_PEDIATRIC_MIN_MM <= pd_mm <= PD_PEDIATRIC_MAX_MM if likely_pediatric else None
        )
        scale_extra["iris_diameter_left_sanity"] = l_iris_sanity
        scale_extra["iris_diameter_right_sanity"] = r_iris_sanity
        scale_extra["ipd_over_iris_ratio_px"] = round(
            pd_px / max(iris_mean_diam_px, 1e-6), 3
        )
        scale_extra["iris_diameter_ipd_ratio_correction"] = iris_ratio_note
        scale_extra["eye_opening_left_px"] = round(eye_l_px, 2)
        scale_extra["eye_opening_right_px"] = round(eye_r_px, 2)
        scale_extra["pd_adult_range_mm"] = f"{PD_ADULT_MIN_MM:g}–{PD_ADULT_MAX_MM:g}"
        scale_extra["pd_in_typical_adult_range"] = (not likely_pediatric) and _pd_in_typical_adult_range_mm(
            pd_mm
        )
        scale_extra["ar_pd_geometry_quality"] = _ar_pd_geometry_quality(
            4.2 <= (pd_px / iris_mean_diam_px) <= 8.5,
            l_iris_sanity,
            r_iris_sanity,
            pd_mm,
            likely_pediatric=likely_pediatric,
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
        left_px = abs(l_ic[0] - mid_x)
        right_px = abs(r_ic[0] - mid_x)
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
        y_pup = (l_ic[1] + r_ic[1]) / 2.0
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

        if ratio_ok:
            pd_note = (
                "PD uses iris centres; when your eyes are level we weight horizontal separation (typical ruler PD). "
                "Scale is mainly iris diameter (~11.77mm) with face-width blend when it agrees. "
                "Geometry: "
                + scale_extra.get("pd_geometry", "")
                + (f"; blend: {scale_extra.get('pd_method', '')}." if scale_extra.get("pd_method") else ".")
                + " For Rx accuracy use an optician or credit-card reference at face depth."
            )
        else:
            pd_note = (
                "Low geometry confidence (unusual iris/IPD ratio or strong head tilt). "
                "Treat PD as approximate; re-capture front-facing at ~60cm or use a reference card."
            )
        if likely_pediatric:
            pd_note += (
                f" Child/small-head mode: the {PD_ADULT_MIN_MM:.0f}–{PD_ADULT_MAX_MM:.0f} mm adult band does not apply; "
                f"rough child range ~{PD_PEDIATRIC_MIN_MM:.0f}–{PD_PEDIATRIC_MAX_MM:.0f} mm. "
                "Webcam PD for children is approximate — use an optician for glasses."
            )
        elif not _pd_in_typical_adult_range_mm(pd_mm):
            pd_note += (
                f" Typical adult PD is often {PD_ADULT_MIN_MM:.0f}–{PD_ADULT_MAX_MM:.0f} mm; "
                "this reading is outside that band — verify framing, distance, or use a reference card."
            )

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
                "pd_note": pd_note,
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

        extra_hf = {k: v for k, v in out["scale"].items() if str(k).startswith("pd_hf")}
        pd_calc_trace = _build_pd_calculation_trace(
            image_w=int(w),
            image_h=int(h),
            left_iris_center_px=(float(l_ic[0]), float(l_ic[1])),
            right_iris_center_px=(float(r_ic[0]), float(r_ic[1])),
            iris_diameter_left_px=float(l_diam),
            iris_diameter_right_px=float(r_diam),
            iris_diameter_mean_px=float(iris_mean_diam_px),
            pd_px_horizontal=float(pd_px_horiz),
            pd_px_euclidean=float(pd_px_eucl),
            eye_dy_px=float(eye_dy),
            face_width_cheek_px=float(fw_px),
            level_ratio=float(level_ratio),
            pd_geometry=str(pd_geom),
            pd_px_used=float(pd_px),
            pd_hint_mm=pd_hint_mm,
            scale_extra=dict(scale_extra),
            pd_mm_before_round=float(pd_mm),
            pd_mm_rounded_half_mm=float(out["mm"]["pd"]),
            ratio_ok=bool(ratio_ok),
            extra_scale_keys=extra_hf,
        )
        _maybe_stdout_pd_trace(pd_calc_trace)
        out["debug"]["pd_calculation_trace"] = pd_calc_trace

        out["_landmark_points_xy"] = pts
        return out
