from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import base64
from io import BytesIO
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

        if result["glasses_detected"]:
            edited_b64 = GlassesService.remove_glasses(image_bytes)

            return {
                "success": True,
                "glasses_detected": True,
                "confidence": result["confidence"],
                "edited_image_base64": edited_b64
            }

        return {
            "success": True,
            "glasses_detected": False,
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    

def prepare_png_base64(edited_bytes: bytes) -> str:
    """
    Ensure the bytes are saved as a proper PNG and encoded as Base64.
    """
    img = Image.open(BytesIO(edited_bytes))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()
 
@router.post("/remove")
async def remove_glasses(image: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await image.read()

        # Call service to remove glasses → returns edited PNG bytes
        edited_bytes = remove_glasses_service(image_bytes)

        # Convert PNG bytes to base64 string
        edited_base64 = base64.b64encode(edited_bytes).decode("utf-8")

        return JSONResponse({
            "success": True,
            "edited_image_base64": edited_base64
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })