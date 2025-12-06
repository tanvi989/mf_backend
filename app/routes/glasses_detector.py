from fastapi import APIRouter, UploadFile, File
from app.services.glasses_service import GlassesService

router = APIRouter(
    prefix="/glasses",
    tags=["Glasses Detection"]
)

@router.post("/detect")
async def detect_glasses(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = GlassesService.detect(image_bytes)

        # If glasses detected → Remove them using Gemini
        if result["glasses_detected"]:
            edited_bytes = GlassesService.remove_glasses(image_bytes)
            return {
                "success": True,
                "glasses_detected": True,
                "confidence": result["confidence"],
                "edited_image_base64": edited_bytes.decode("latin1")  # send raw bytes
            }

        return {
            "success": True,
            "glasses_detected": False,
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
