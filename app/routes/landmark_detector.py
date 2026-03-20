import json
from typing import Any, Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.db.virtual_tryon_repo import update_measurements
from app.services.age_service import predict_age_from_bytes
from app.services.credit_card_measurement_service import CreditCardMeasurementService
from app.services.eyewear_insights_service import build_eyewear_insights
from app.services.gender_service import predict_gender_from_bytes
from app.services.iris_landmark_service import IrisLandmarkService

router = APIRouter(prefix="/landmarks", tags=["Landmark Detection"])


def _json_safe(value: Any) -> Any:
    """Ensure Motor + client JSON never see numpy / exotic types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@router.post("/detect")
async def detect_landmarks(
    file: UploadFile = File(...),
    guest_id: str = "temp_guest",
    session_id: str = "temp_session",
    pd_hint_mm: Optional[float] = Form(None),
    gender_image: Optional[UploadFile] = File(None),
    client_capture: Optional[str] = Form(None),
):
    try:
        image_bytes = await file.read()

        gender_bytes: Optional[bytes] = None
        if gender_image is not None:
            try:
                g = await gender_image.read()
                if g and len(g) > 100:
                    gender_bytes = g
            except Exception:
                gender_bytes = None

        hint = pd_hint_mm
        if hint is not None:
            try:
                hint = float(hint)
            except (TypeError, ValueError):
                hint = None
            if hint is not None and not (48.0 <= hint <= 80.0):
                hint = None

        result = IrisLandmarkService.detect_landmarks(image_bytes, pd_hint_mm=hint)
        points = result.pop("_landmark_points_xy", None)

        client_parsed: Optional[dict] = None
        if client_capture and isinstance(client_capture, str) and client_capture.strip():
            try:
                raw = client_capture.strip()[:20000]
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    client_parsed = loaded
            except Exception:
                client_parsed = None
        if client_parsed is not None:
            result["client_capture"] = _json_safe(client_parsed)

        # Prefer natural face for gender when glasses were removed (processed image is often PNG/AI-edited).
        gsrc = gender_bytes if gender_bytes is not None else image_bytes
        try:
            gender = predict_gender_from_bytes(
                gsrc,
                landmark_points=points if gsrc is image_bytes else None,
            )
        except Exception as ge:
            gender = {
                "label": "unknown",
                "confidence": 0.0,
                "low_confidence": True,
                "model": None,
                "error": str(ge)[:200],
            }
        gender = _json_safe(gender)
        result["gender"] = gender

        try:
            eyewear = build_eyewear_insights(
                result["mm"],
                result.get("scale") or {},
                str(result.get("face_shape") or ""),
            )
            try:
                eyewear["age_estimate"] = _json_safe(
                    predict_age_from_bytes(
                        gsrc,
                        landmark_points=points if gsrc == image_bytes else None,
                    )
                )
            except Exception as ae:
                eyewear["age_estimate"] = {
                    "bucket": "unknown",
                    "confidence": 0.0,
                    "low_confidence": True,
                    "model": None,
                    "error": str(ae)[:200],
                }
        except Exception as ex:
            eyewear = {
                "face_width_bucket": "unknown",
                "fit_hint": "review",
                "warnings": [f"Eyewear bundle failed to build: {str(ex)[:200]}"],
                "style_tips": [],
                "disclaimer": "Shopping hints only — not a prescription.",
                "age_estimate": {
                    "bucket": "unknown",
                    "confidence": 0.0,
                    "low_confidence": True,
                    "model": None,
                    "error": "skipped_after_insights_failure",
                },
            }
        eyewear = _json_safe(eyewear)
        result["eyewear"] = eyewear

        await update_measurements(
            guest_id=guest_id,
            session_id=session_id,
            mm=_json_safe(result["mm"]),
            face_shape=str(result["face_shape"]),
            gender=gender,
            eyewear=eyewear,
            client_capture=result.get("client_capture"),
        )

        return {"success": True, "landmarks": _json_safe(result)}

    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/credit-card")
async def measure_with_credit_card(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = CreditCardMeasurementService.process(image_bytes)
        return {"success": True, "landmarks": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
