from fastapi import APIRouter, UploadFile, File
from services.landmark_service import LandmarkService

router = APIRouter(
    prefix="/landmarks",
    tags=["Landmark Detection"]
)

@router.post("/detect")
async def detect_landmarks(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = LandmarkService.detect_landmarks(image_bytes)

        return {
            "success": True,
            "landmarks": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
