"""Eyewear e-commerce hints: sizing, frame SKUs, lens height, segment proxy, ML status."""

from __future__ import annotations

from typing import Any

# Keep in sync with perfect-fit-cam FramesTab FRAMES[] (mm).
_FRAME_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "frame_1",
        "name": "Pink Cat-Eye",
        "width_mm": 127,
        "nose_bridge_mm": 15,
        "lens_width_mm": 50,
    },
    {
        "id": "frame_2",
        "name": "Blue Round",
        "width_mm": 122,
        "nose_bridge_mm": 18,
        "lens_width_mm": 44,
    },
    {
        "id": "frame_3",
        "name": "Black Aviator",
        "width_mm": 141,
        "nose_bridge_mm": 18,
        "lens_width_mm": 55,
    },
)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _lens_height_band(face_height_mm: float, face_ratio: float) -> dict[str, Any]:
    """Suggest lens height (B) range from face proportions — heuristic only."""
    fh = face_height_mm
    fr = face_ratio
    if fh >= 156 or fr < 0.78:
        return {
            "label": "taller / longer face",
            "suggested_lens_height_mm_min": 40,
            "suggested_lens_height_mm_max": 52,
            "explanation": (
                "Longer face height often suits slightly taller lenses; tiny round frames can look undersized."
            ),
        }
    if fh <= 128 or fr > 0.92:
        return {
            "label": "compact face",
            "suggested_lens_height_mm_min": 32,
            "suggested_lens_height_mm_max": 42,
            "explanation": (
                "Shorter face height often suits modest lens depth; very tall lenses can overwhelm proportions."
            ),
        }
    return {
        "label": "average proportions",
        "suggested_lens_height_mm_min": 36,
        "suggested_lens_height_mm_max": 48,
        "explanation": (
            "Mid-range lens heights (roughly high 30s–mid 40s mm) usually work; refine with bridge fit and style."
        ),
    }


def _frame_width_fit(
    frame_width_mm: float, band_min: float, band_max: float
) -> tuple[str, str]:
    if frame_width_mm < band_min - 6:
        return (
            "narrow",
            f"This frame ({frame_width_mm:.0f} mm) is slimmer than the {band_min:.0f}–{band_max:.0f} mm band "
            "suggested from your cheek width — may feel tight at the temples or look small on the face.",
        )
    if frame_width_mm > band_max + 8:
        return (
            "wide",
            f"This frame ({frame_width_mm:.0f} mm) is wider than the typical band for your face — "
            "may look oversized or slide if the bridge does not sit well.",
        )
    return (
        "good",
        f"Total frame width ({frame_width_mm:.0f} mm) sits near your suggested {band_min:.0f}–{band_max:.0f} mm range.",
    )


def _bridge_note(
    frame_bridge: float, nose_proxy_mm: float
) -> tuple[str, str]:
    d = abs(frame_bridge - nose_proxy_mm)
    if d <= 3:
        return (
            "good",
            f"Listed bridge ({frame_bridge:.0f} mm) is close to your nose-width proxy (~{nose_proxy_mm:.0f} mm).",
        )
    if d <= 6:
        return (
            "review",
            f"Bridge ({frame_bridge:.0f} mm) differs from your proxy (~{nose_proxy_mm:.0f} mm) by ~{d:.0f} mm — "
            "adjustable nose pads or try-on can still work.",
        )
    return (
        "wide_mismatch",
        f"Bridge ({frame_bridge:.0f} mm) is quite different from your proxy (~{nose_proxy_mm:.0f} mm) — "
        "prioritize fitting or adjustable bridges.",
    )


def build_eyewear_insights(
    mm: dict,
    scale: dict,
    face_shape: str,
) -> dict[str, Any]:
    fw = _f(mm.get("face_width"))
    fh = _f(mm.get("face_height"))
    fr = _f(mm.get("face_ratio"))
    pd_t = _f(mm.get("pd"))
    pl = _f(mm.get("pd_left"))
    pr = _f(mm.get("pd_right"))
    pd_hf = mm.get("pd_hf")
    pd_hf_f = _f(pd_hf) if pd_hf is not None else None

    nb_l = _f(mm.get("nose_bridge_left"))
    nb_r = _f(mm.get("nose_bridge_right"))
    nose_proxy = (nb_l + nb_r) / 2.0 if (nb_l + nb_r) > 1e-6 else 18.0

    eye_v = mm.get("eye_vertical_position_ratio")
    try:
        eye_v_f = float(eye_v) if eye_v is not None else None
    except (TypeError, ValueError):
        eye_v_f = None

    seg_mm = mm.get("segment_height_proxy_mm")
    try:
        seg_f = float(seg_mm) if seg_mm is not None else None
    except (TypeError, ValueError):
        seg_f = None

    chin_ratio = mm.get("chin_to_face_width_ratio")
    try:
        chin_r_f = float(chin_ratio) if chin_ratio is not None else None
    except (TypeError, ValueError):
        chin_r_f = None

    if fw < 80:
        face_width_bucket = "unknown"
    elif fw < 130:
        face_width_bucket = "narrow"
    elif fw <= 145:
        face_width_bucket = "medium"
    else:
        face_width_bucket = "wide"

    band_min = round(max(fw - 4, 0), 1)
    band_max = round(fw + 10, 1)

    pd_rel = (scale or {}).get("pd_reliability") or "unknown"
    pd_method = (scale or {}).get("pd_method") or ""

    mono_asym = abs(pl - pr) if pd_t > 1e-6 else 0.0
    nose_asym = abs(nb_l - nb_r)

    warnings: list[str] = []
    if mono_asym >= 2.5:
        warnings.append(
            f"Monocular PD differs by {mono_asym:.1f} mm — verify with an optician if ordering progressive lenses."
        )
    if nose_asym >= 6.0:
        warnings.append(
            "Nose-bridge distances left vs right are uneven — frame sit may need adjustment."
        )
    if pd_rel == "low":
        warnings.append("PD geometry confidence is low — retake a straight-on photo or use in-person PD.")
    if pd_hf_f is not None and abs(pd_t - pd_hf_f) >= 4.0:
        warnings.append(
            f"Primary PD ({pd_t:.1f} mm) and HF landmark PD ({pd_hf_f:.1f} mm) disagree — treat as approximate."
        )

    tips: list[str] = []
    fs = (face_shape or "").lower().strip()
    if fs == "round":
        tips.append("Angular or rectangular frames often add structure to rounder faces.")
    elif fs == "square":
        tips.append("Round or oval frames can soften strong jawlines.")
    elif fs == "oval":
        tips.append("Most frame shapes work; use nose bridge and PD to narrow size.")
    elif fs == "rectangle":
        tips.append("Try frames with depth or decorative brow lines to balance face length.")
    elif fs == "heart":
        tips.append("Light or rimless lower rims can balance a wider forehead.")
    else:
        tips.append("Use face width and PD to pick total frame width first, then bridge fit.")

    if face_width_bucket == "narrow":
        tips.append("Filter for narrow / small sizing; watch total frame width vs suggested range.")
    elif face_width_bucket == "wide":
        tips.append("Look for wide or Asian-fit / large sizing lines if temples pinch.")

    if eye_v_f is not None:
        if eye_v_f > 0.43:
            tips.append(
                "Eyes sit relatively low in the face — slightly taller lenses or a higher bridge feel can balance."
            )
        elif eye_v_f < 0.34:
            tips.append(
                "Eyes sit relatively high — very heavy top rims may crowd the brow; try lighter brow lines."
            )

    if chin_r_f is not None:
        if chin_r_f < 0.78:
            tips.append("Narrower chin vs cheek width — tapered or cat-eye sweeps can echo jaw taper.")
        elif chin_r_f > 0.92:
            tips.append("Wide chin vs cheek — squared or geometric bottoms can align with jaw width.")

    lens_height = _lens_height_band(fh, fr)

    segment_block: dict[str, Any] = {
        "pupil_to_lower_lid_proxy_mm": seg_f,
        "note": (
            "Approximate vertical span from iris centre to lower eyelid landmarks (single front photo). "
            "True segment height for progressives is measured from the fitting cross — not a substitute for an optician."
        ),
        "progressives_disclaimer": (
            "For multifocal / progressive orders, use a professional fitting height; this app value is for education only."
        ),
    }

    capture_quality = {
        "pd_geometry": pd_rel,
        "eyes_open_frontal_hint": (
            "Your capture passed in-app alignment checks; for best results keep eyes level and well lit."
            if pd_rel == "high"
            else "Retake with a straight, front-facing pose and steady hold so PD and lid metrics are reliable."
        ),
    }

    frame_fits: list[dict[str, Any]] = []
    for f in _FRAME_CATALOG:
        wf, wx = _frame_width_fit(float(f["width_mm"]), band_min, band_max)
        bf, bx = _bridge_note(float(f["nose_bridge_mm"]), nose_proxy)
        if wf == "good" and bf == "good":
            overall = "good fit (on paper)"
        elif wf == "good" or bf == "good":
            overall = "mixed — check try-on"
        else:
            overall = "review sizing"
        frame_fits.append(
            {
                "id": f["id"],
                "name": f["name"],
                "frame_total_width_mm": f["width_mm"],
                "frame_nose_bridge_mm": f["nose_bridge_mm"],
                "frame_lens_width_mm": f["lens_width_mm"],
                "width_vs_face": wf,
                "width_explanation": wx,
                "bridge_vs_estimate": bf,
                "bridge_explanation": bx,
                "overall_label": overall,
            }
        )

    features_status = {
        "frame_fit_per_catalog_sku": "computed",
        "face_width_category": "computed",
        "lens_height_heuristic": "computed",
        "segment_height_proxy": "computed" if seg_f is not None else "unavailable_or_out_of_range",
        "eye_vertical_position": "computed" if eye_v_f is not None else "unavailable",
        "jaw_chin_ratio": "computed" if chin_r_f is not None else "unavailable",
        "style_rules_face_shape": "computed",
        "gender_ml": "returned_separately_in_landmarks.gender",
        "age_band_ml": "returned_in_eyewear.age_estimate",
        "face_parsing_hair_vto": "not_enabled",
        "emotion_attention_gate": "partially_via_frontend_pose_only",
        "monocular_depth_bridge": "not_enabled",
        "face_embedding_similar_products": "not_enabled_privacy",
    }

    fit_hint = "good"
    if warnings:
        fit_hint = "review"

    return {
        "face_width_bucket": face_width_bucket,
        "face_width_bucket_recommendation": (
            f"Merch copy: “Recommended for {face_width_bucket} faces” when filtering SKUs "
            "(narrow ~–129 mm cheek, medium 130–145 mm, wide 146+ mm vs our estimator)."
        ),
        "face_width_mm": round(fw, 1) if fw > 0 else None,
        "face_height_mm": round(fh, 1) if fh > 0 else None,
        "suggested_frame_total_width_mm": {"min": band_min, "max": band_max},
        "lens_height_guidance": lens_height,
        "segment_height": segment_block,
        "eye_vertical_position_ratio": eye_v_f,
        "chin_to_face_width_ratio": chin_r_f,
        "pd_reliability": pd_rel,
        "pd_blend_method": pd_method or None,
        "monocular_asymmetry_mm": round(mono_asym, 2) if pd_t > 1e-6 else None,
        "nose_bridge_asymmetry_mm": round(nose_asym, 2),
        "nose_bridge_proxy_mm": round(nose_proxy, 1),
        "catalog_frame_fit": frame_fits,
        "capture_quality": capture_quality,
        "features_status": features_status,
        "fit_hint": fit_hint,
        "warnings": warnings,
        "style_tips": tips,
        "disclaimer": (
            "Shopping hints only — not a prescription or medical device. "
            "Final PD, segment height, and frame fit should be confirmed by an optician for Rx orders."
        ),
    }
