from fastapi import APIRouter, UploadFile, File, Form
from app.db.virtual_tryon_repo import get_virtual_tryon_by_session, save_selected_frame
from pydantic import BaseModel
from app.services.gcs_service import upload_image_and_get_url

virtual_tryon = APIRouter(
    prefix="/virtual-tryon",
    tags=["Virtual Tryon"]
)

@virtual_tryon.get("/session")
async def get_session_details(
    guest_id: str,
    session_id: str
):
    try:
        data = await get_virtual_tryon_by_session(guest_id, session_id)

        if not data:
            return {
                "success": False,
                "error": "Session not found"
            }

        return {
            "success": True,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@virtual_tryon.post("/select-frame")
async def select_frame(
    guest_id: str = Form(...),
    session_id: str = Form(...),
    frame_id: str = Form(...),
    frame_name: str = Form(...),
    selected_frame_image: UploadFile = File(...)
):
    try:
        image_bytes = await selected_frame_image.read()

        # ✅ Upload selected frame image
        frame_upload = upload_image_and_get_url(
            file_bytes=image_bytes,
            guest_id=guest_id,
            session_id=session_id,
            stage="selected_frame",
            ext=selected_frame_image.filename.split(".")[-1]
        )

        success = await save_selected_frame(
            guest_id=guest_id,
            session_id=session_id,
            frame_id=frame_id,
            frame_name=frame_name,
            frame_image=frame_upload["signed_url"]  # or ["url"]
        )

        if not success:
            return {
                "success": False,
                "error": "Session not found"
            }

        return {
            "success": True,
            "frame_image": frame_upload["signed_url"],
            "message": "Frame selected successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
