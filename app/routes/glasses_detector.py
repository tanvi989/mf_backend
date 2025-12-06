from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.glasses_service import GlassesService
from app.services.glasses_removal import remove_glasses_service
router = APIRouter(
    prefix="/glasses",
    tags=["Glasses Detection", "Glasses Removal"]
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

@router.post("/remove")
async def remove_glasses(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        edited_base64 = remove_glasses_service(image_bytes)

        return JSONResponse({
            "success": True,
            "edited_image_base64": edited_base64
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)