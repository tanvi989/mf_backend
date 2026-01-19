from datetime import datetime
from app.db.mongo import virtual_tryons

async def update_original_image(
    guest_id: str,
    session_id: str,
    image_data: dict
):
    await virtual_tryons.update_one(
        {"guest_id": guest_id, "session_id": session_id},
        {
            "$set": {
                "images.original": {
                    **image_data,
                    "saved_at": datetime.utcnow()
                }
            },
            "$currentDate": {"updated_at": True}
        }
    )
