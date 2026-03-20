"""Secondary PD estimate: InsightFace 2d106det ONNX from Hugging Face (coordinate regression).

IPD is taken from 2d106 landmarks in an upper-face band (bicentric split), with a ratio-based pair
fallback — not fixed pupil indices (104–105 are not bilateral pupils in this ONNX).
mm/px uses the same ~11.77 mm iris ruler as the primary pipeline.
"""

from __future__ import annotations

import math
import threading
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

# Keep in sync with iris_landmark_service.IRIS_DIAMETER_MM (single mm/px ruler).
_IRIS_DIAMETER_MM = 11.77

_PD_HF_DIR = Path(__file__).resolve().parent.parent / "models" / "pd_hf_onnx"
_PD_HF_FILE = _PD_HF_DIR / "2d106det.onnx"
_PD_HF_URL = (
    "https://huggingface.co/menglaoda/_insightface/resolve/main/2d106det.onnx"
)

_PD_HF_LOCK = threading.Lock()
_PD_HF_SESSION: ort.InferenceSession | None = None


def _ensure_model() -> None:
    _PD_HF_DIR.mkdir(parents=True, exist_ok=True)
    if _PD_HF_FILE.exists():
        sz = _PD_HF_FILE.stat().st_size
        if sz >= 1_000_000:
            with open(_PD_HF_FILE, "rb") as fh:
                head = fh.read(40)
            if not head.startswith(b"version https://git-lfs"):
                return
        try:
            _PD_HF_FILE.unlink()
        except OSError:
            pass
    with urllib.request.urlopen(_PD_HF_URL, timeout=180) as r:  # noqa: S310
        _PD_HF_FILE.write_bytes(r.read())


def _session() -> ort.InferenceSession:
    global _PD_HF_SESSION
    with _PD_HF_LOCK:
        if _PD_HF_SESSION is None:
            _ensure_model()
            try:
                _PD_HF_SESSION = ort.InferenceSession(
                    str(_PD_HF_FILE), providers=["CPUExecutionProvider"]
                )
            except Exception:
                try:
                    _PD_HF_FILE.unlink()
                except OSError:
                    pass
                _ensure_model()
                _PD_HF_SESSION = ort.InferenceSession(
                    str(_PD_HF_FILE), providers=["CPUExecutionProvider"]
                )
        return _PD_HF_SESSION


def _insightface_crop(
    img_bgr: np.ndarray, center: tuple[float, float], output_size: int, scale: float, rotation: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    scale_ratio = scale
    rot = float(rotation) * math.pi / 180.0
    t1 = SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = SimilarityTransform(translation=(-1.0 * cx, -1.0 * cy))
    t3 = SimilarityTransform(rotation=rot)
    t4 = SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    m = t.params[0:2, :].astype(np.float32)
    cropped = cv2.warpAffine(
        img_bgr, m, (output_size, output_size), borderValue=0.0
    )
    return cropped, m


def _trans_points_2d(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pts, dtype=np.float32)
    for i in range(pts.shape[0]):
        p = np.array([pts[i, 0], pts[i, 1], 1.0], dtype=np.float32)
        q = m @ p
        out[i] = q[0:2]
    return out


def _bbox_from_mediapipe_pts(
    mediapipe_xy: list[tuple[int, int]], n: int = 468
) -> tuple[float, float, float, float]:
    n = min(n, len(mediapipe_xy))
    if n < 10:
        raise ValueError("Too few landmarks for bbox")
    xs = [float(mediapipe_xy[i][0]) for i in range(n)]
    ys = [float(mediapipe_xy[i][1]) for i in range(n)]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2, y2


def _eye_band_indices(pred_img: np.ndarray, yrel_lo: float, yrel_hi: float) -> list[int]:
    ymin = float(pred_img[:, 1].min())
    ymax = float(pred_img[:, 1].max())
    fh = max(ymax - ymin, 1e-3)
    return [
        i
        for i in range(106)
        if yrel_lo <= (float(pred_img[i, 1]) - ymin) / fh <= yrel_hi
    ]


def _pd_px_from_eye_band_bicentric(
    pred_img: np.ndarray,
    face_mid_x: float,
    yrel_lo: float = 0.18,
    yrel_hi: float = 0.50,
    min_per_side: int = 2,
    midline_strip_frac: float | None = None,
) -> tuple[float, str]:
    """Horizontal separation of left vs right eye-band landmark groups.

    Without a strip, inner-canthus / nose-adjacent points sit near the midline and pull
    each side's mean *inward*, underestimating IPD vs iris-centre PD (~2–4 mm typical).
    ``midline_strip_frac`` removes landmarks within ``frac * face_width_2d106`` of
    ``face_mid_x`` so centroids emphasize *lateral* eye contours (closer to clinical IPD).
    """
    band_idx = _eye_band_indices(pred_img, yrel_lo, yrel_hi)
    if len(band_idx) < 4:
        return 0.0, "eye_band_too_few_points"

    if midline_strip_frac is not None and midline_strip_frac > 1e-6:
        xmin = float(pred_img[:, 0].min())
        xmax = float(pred_img[:, 0].max())
        fw106 = max(xmax - xmin, 1e-3)
        half_w = float(midline_strip_frac) * fw106
        band_idx = [
            i
            for i in band_idx
            if abs(float(pred_img[i, 0]) - float(face_mid_x)) >= half_w
        ]
        if len(band_idx) < 4:
            return 0.0, "eye_band_mid_strip_too_few"

    left_i = [i for i in band_idx if float(pred_img[i, 0]) < face_mid_x]
    right_i = [i for i in band_idx if float(pred_img[i, 0]) >= face_mid_x]
    if len(left_i) < min_per_side or len(right_i) < min_per_side:
        return 0.0, "eye_band_split_empty"

    lx = float(np.mean(pred_img[left_i, 0]))
    rx = float(np.mean(pred_img[right_i, 0]))
    tag = "eye_band_lateral" if midline_strip_frac else "eye_band_bicentric"
    return abs(rx - lx), tag


def _ratio_in_range(pd_px_h: float, iris_mean_diam_px: float, lo: float, hi: float) -> bool:
    if pd_px_h < 1e-6:
        return False
    r = pd_px_h / max(iris_mean_diam_px, 1e-6)
    return lo <= r <= hi


def _pd_px_from_best_aligned_pair(
    pred_img: np.ndarray,
    iris_mean_diam_px: float,
    primary_pd_px: float,
) -> tuple[float, str]:
    """Pick the landmark pair in the eye band whose IPD/iris ratio is closest to MediaPipe's.

    Indices 104–105 are *not* reliable pupils for this ONNX; this search finds a consistent chord.
    """
    if primary_pd_px <= 1e-6 or iris_mean_diam_px < 1e-6:
        return 0.0, "no_primary_geometry"
    target_ratio = primary_pd_px / iris_mean_diam_px
    ymin = float(pred_img[:, 1].min())
    ymax = float(pred_img[:, 1].max())
    fh = max(ymax - ymin, 1e-3)

    best_h = 0.0
    best_score = float("inf")
    for i in range(106):
        for j in range(i + 1, 106):
            yi = (float(pred_img[i, 1]) - ymin) / fh
            yj = (float(pred_img[j, 1]) - ymin) / fh
            if not (0.18 <= yi <= 0.52 and 0.18 <= yj <= 0.52):
                continue
            if abs(float(pred_img[i, 1] - pred_img[j, 1])) > 0.10 * fh:
                continue
            h = abs(float(pred_img[i, 0] - pred_img[j, 0]))
            r = h / iris_mean_diam_px
            if not (3.0 <= r <= 10.0):
                continue
            score = abs(r - target_ratio)
            if score < best_score:
                best_score = score
                best_h = h

    if best_h <= 1e-6:
        return 0.0, "no_valid_pair"
    return best_h, "ratio_matched_pair"


def estimate_pd_hf_from_mediapipe_crop(
    rgb: np.ndarray,
    mediapipe_pts: list[tuple[int, int]],
    iris_mean_diam_px: float,
    primary_pd_mm: float,
    primary_pd_px: float,
    primary_left_mm: float,
    primary_right_mm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (mm_fragment, scale_fragment) for merging into landmark detect output."""
    scale: dict[str, Any] = {
        "pd_hf_model": "insightface_2d106det · Hugging Face ONNX",
        "pd_hf_provenance": _PD_HF_URL,
        "pd_hf_error": None,
        "pd_hf_note": None,
        "pd_hf_method": None,
        "pd_hf_px_horizontal": None,
        "pd_hf_ratio_iris": None,
        "pd_hf_delta_mm": None,
    }
    mm: dict[str, Any] = {
        "pd_hf": None,
        "pd_hf_left": None,
        "pd_hf_right": None,
    }

    if iris_mean_diam_px < 1e-3:
        scale["pd_hf_error"] = "iris diameter px too small"
        return mm, scale

    try:
        x1, y1, x2, y2 = _bbox_from_mediapipe_pts(mediapipe_pts)
    except Exception as e:
        scale["pd_hf_error"] = str(e)[:200]
        return mm, scale

    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    face_mid_x = float(center[0])
    out_size = 192
    scale_f = out_size / (max(w, h) * 1.5)

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    aimg, m_forward = _insightface_crop(img_bgr, center, out_size, scale_f, 0.0)

    sess = _session()
    inp = sess.get_inputs()[0]
    oname = sess.get_outputs()[0].name
    blob = cv2.dnn.blobFromImage(
        aimg,
        scalefactor=1.0 / 128.0,
        size=(out_size, out_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )
    pred = sess.run([oname], {inp.name: blob})[0][0]
    pred = np.asarray(pred, dtype=np.float32).reshape((-1, 2))
    if pred.shape[0] > 106:
        pred = pred[-106:, :]
    if pred.shape[0] < 106:
        scale["pd_hf_error"] = f"unexpected landmark count {pred.shape[0]}"
        return mm, scale

    pred[:, 0:2] += 1.0
    pred[:, 0:2] *= float(out_size // 2)
    m_inv = cv2.invertAffineTransform(m_forward)
    pred_img = _trans_points_2d(pred, m_inv)

    mid_mediapipe = float(face_mid_x)
    mid_landmarks = float(np.median(pred_img[:, 0]))

    pd_px_h = 0.0
    method_parts: list[str] = []
    R_LO, R_HI = 3.05, 9.85  # accept slightly wider than primary sanity; avoids false "unavailable"

    # Try many band / midline / strip settings. Pure "bicentric" leans low (inner eye pulls means in);
    # aggressive strips lean high (only outer contour left). Among configs with sane IPD/iris ratio,
    # pick pixel IPD closest to MediaPipe's horizontal iris chord — same scene, not copying mm output.
    strip_schedule = (0.055, 0.07, 0.04, 0.085, None)
    y_schedule = ((0.22, 0.46), (0.20, 0.48), (0.18, 0.50), (0.14, 0.56))
    mids = ((mid_mediapipe, "mp_mid"), (mid_landmarks, "lm_mid"))

    candidates: list[tuple[float, float, str]] = []
    for strip in strip_schedule:
        for mid_x, mid_label in mids:
            for ylo, yhi in y_schedule:
                d, mname = _pd_px_from_eye_band_bicentric(
                    pred_img,
                    mid_x,
                    ylo,
                    yhi,
                    min_per_side=2,
                    midline_strip_frac=strip,
                )
                if not _ratio_in_range(d, iris_mean_diam_px, R_LO, R_HI):
                    continue
                s = "" if strip is None else f",strip={strip:g}"
                tag = f"{mname}({mid_label},{ylo}-{yhi}{s})"
                dist = abs(d - float(primary_pd_px))
                candidates.append((dist, d, tag))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        _, pd_px_h, best_tag = candidates[0]
        method_parts.append(f"{best_tag}|nearest_mp_px")

    scale["pd_hf_method"] = ";".join(method_parts) if method_parts else None

    if pd_px_h < 1e-3 or not _ratio_in_range(pd_px_h, iris_mean_diam_px, R_LO, R_HI):
        pd_px_h_fb, method_fb = _pd_px_from_best_aligned_pair(
            pred_img, iris_mean_diam_px, primary_pd_px
        )
        scale["pd_hf_method"] = (
            f"{scale['pd_hf_method'] or 'bicentric_fail'}|{method_fb}"
        )
        if pd_px_h_fb > 1e-6 and _ratio_in_range(
            pd_px_h_fb, iris_mean_diam_px, R_LO, R_HI
        ):
            pd_px_h = pd_px_h_fb

    pd_ratio = pd_px_h / max(iris_mean_diam_px, 1e-6)
    primary_ratio = primary_pd_px / max(iris_mean_diam_px, 1e-6) if primary_pd_px > 0 else 0.0

    if pd_px_h < 1e-3 or not (3.0 <= pd_ratio <= 10.0):
        scale["pd_hf_error"] = (
            f"2d106 IPD/iris ratio out of range ({pd_ratio:.2f}) or geometry failed; "
            f"methods tried: {scale.get('pd_hf_method')}"
        )
        scale["pd_hf_px_horizontal"] = round(pd_px_h, 3) if pd_px_h > 1e-6 else None
        scale["pd_hf_ratio_iris"] = round(pd_ratio, 3) if pd_px_h > 1e-6 else None
        return mm, scale

    s_iris = _IRIS_DIAMETER_MM / iris_mean_diam_px
    pd_hf_mm = pd_px_h * s_iris
    pd_hf_mm = round(float(pd_hf_mm) * 2.0) / 2.0

    mm["pd_hf"] = pd_hf_mm
    scale["pd_hf_px_horizontal"] = round(pd_px_h, 3)
    scale["pd_hf_ratio_iris"] = round(pd_ratio, 3)
    scale["pd_hf_delta_mm"] = (
        round(abs(primary_pd_mm - pd_hf_mm), 2) if math.isfinite(primary_pd_mm) else None
    )
    scale["pd_hf_note"] = (
        "2d106 uses eye-region landmarks, not iris centres. Naive left/right means can read low (inner eye pulls inward) "
        "or high (only outer contour left). We try several bands and midline strips, keep only plausible IPD/iris ratios, "
        "then choose the 2d106 chord in pixels closest to MediaPipe’s iris IPD (same photo) before converting to mm — "
        "so HF tracks the same inter-eye span without copying the primary’s blended millimetres. For dispensing, prefer an in-person PD."
    )

    if primary_pd_mm > 1e-6:
        rl = primary_left_mm / primary_pd_mm
        rr = primary_right_mm / primary_pd_mm
        mm["pd_hf_left"] = round(float(pd_hf_mm * rl) * 2.0) / 2.0
        mm["pd_hf_right"] = round(float(pd_hf_mm * rr) * 2.0) / 2.0

    # Hint if HF agrees with iris geometry but disagrees with blended primary (face prior / hint).
    if primary_ratio > 0 and abs(pd_ratio - primary_ratio) > 0.55:
        scale["pd_hf_note"] += " Primary vs HF IPD/iris ratios differ — check frontal pose and distance."

    return mm, scale
