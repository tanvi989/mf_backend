from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from app.services.glasses_service import GlassesService
from app.services.glasses_removal import remove_glasses_service
from app.db.virtual_tryon_repo import update_tryon
from app.db.tryon_insert import insert_on_detection
from app.services.gcs_service import upload_image_and_get_url

router = APIRouter(
    prefix="/glasses",
    tags=["Glasses Detection"]
)

# ------------------------
# DETECT API
# ------------------------
@router.post("/detect")
async def detect_glasses(
    file: UploadFile = File(...),
    guest_id: str = "temp_guest",
    session_id: str = "temp_session"
):
    try:
        image_bytes = await file.read()

        # Upload ORIGINAL image
        original_upload = upload_image_and_get_url(
            file_bytes=image_bytes,
            guest_id=guest_id,
            session_id=session_id,
            stage="original",
            ext="jpg"
        )
        # Detect glasses
        result = GlassesService.detect(image_bytes)

        await insert_on_detection(
            guest_id=guest_id,
            session_id=session_id,
            glasses_detected=result["glasses_detected"],
            confidence=result["confidence"]
        )

        # 4️⃣ Update original image path
        await update_tryon(
            guest_id=guest_id,
            session_id=session_id,
            update_data={
                "images.original.bucket_path": original_upload["signed_url"]}
        )

        return {
            "success": True,
            "glasses_detected": result["glasses_detected"],
            "confidence": result["confidence"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------
# REMOVE API
# ------------------------
@router.post("/remove")
async def remove_glasses(
    image: UploadFile = File(...),
    guest_id: str = "temp_guest",
    session_id: str = "temp_session"
):
    try:
        image_bytes = await image.read()

        # Remove glasses
        edited_bytes = remove_glasses_service(image_bytes)

        # Upload glasses-removed image
        removed_upload = upload_image_and_get_url(
            file_bytes=edited_bytes,
            guest_id=guest_id,
            session_id=session_id,
            stage="glasses_removed",
            ext="jpg"
        )

        # Update try-on document with removed image
        await update_tryon(
            guest_id=guest_id,
            session_id=session_id,
            update_data={
                "images.glasses_removed.bucket_path": removed_upload["signed_url"]
            }
        )

        # Return base64 PNG
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