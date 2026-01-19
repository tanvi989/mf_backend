from fastapi import APIRouter, UploadFile, File
# from app.services.landmark_service import LandmarkService
from app.services.iris_landmark_service import IrisLandmarkService
from app.services.credit_card_measurement_service import CreditCardMeasurementService
from app.db.virtual_tryon_repo import update_measurements
router = APIRouter(
    prefix="/landmarks",
    tags=["Landmark Detection"]
)

@router.post("/detect")
async def detect_landmarks(
    file: UploadFile = File(...),
    guest_id: str = "temp_guest",
    session_id: str = "temp_session"
):
    try:
        image_bytes = await file.read()

        # Detect landmarks
        result = IrisLandmarkService.detect_landmarks(image_bytes)

        # Persist required measurements
        await update_measurements(
            guest_id=guest_id,
            session_id=session_id,
            mm=result["mm"],
            face_shape=result["face_shape"]
        )

        return {
            "success": True,
            "landmarks": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
  

@router.post("/credit-card")
async def measure_with_credit_card(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = CreditCardMeasurementService.process(image_bytes)

        return {
            "success": True,
            "landmarks": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }