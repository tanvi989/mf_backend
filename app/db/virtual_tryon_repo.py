from datetime import datetime
from typing import Optional

from fastapi import UploadFile, File
from app.db.mongo import virtual_tryons

# 🔹 INSERT (ONLY ONCE)
async def insert_tryon(
    guest_id: str,
    session_id: str,
    detection: dict
):
    await virtual_tryons.insert_one({
        "guest_id": guest_id,
        "session_id": session_id,

        "images": {},
        "measurements": {},

        "glasses": detection,

        "status": {
            "detected": True,
            "glasses_removed": False,
            "measurements_done": False
        },

        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })


# 🔹 UPDATE HELPERS
async def update_tryon(
    guest_id: str,
    session_id: str,
    update_data: dict
):
    await virtual_tryons.update_one(
        {"guest_id": guest_id, "session_id": session_id},
        {
            "$set": update_data,
            "$currentDate": {"updated_at": True}
        }
    )


async def update_measurements(
    guest_id: str,
    session_id: str,
    mm: dict,
    face_shape: str,
    gender: Optional[dict] = None,
    eyewear: Optional[dict] = None,
    client_capture: Optional[dict] = None,
):
    now = datetime.utcnow()

    fields = {
        "measurements.mm": mm,
        "measurements.face_shape": face_shape,
        "status.measurements_done": True,
        "updated_at": now,
    }
    if gender is not None:
        fields["measurements.gender"] = gender
    if eyewear is not None:
        fields["measurements.eyewear"] = eyewear
    if client_capture is not None:
        fields["measurements.client_capture"] = client_capture

    await virtual_tryons.update_one(
        {"guest_id": guest_id, "session_id": session_id},
        {"$set": fields},
        upsert=False,  # detect API must run first
    )

async def get_virtual_tryon_by_session(
    guest_id: str,
    session_id: str
):
    return await virtual_tryons.find_one(
        {
            "guest_id": guest_id,
            "session_id": session_id
        },
        {
            "_id": 0  # hide mongo id
        }
    )

async def save_selected_frame(
    guest_id: str,
    session_id: str,
    frame_id: str,
    frame_name: str,
    frame_dims: dict,
    fitting_height: float,
    frame_image_url: str
):
    now = datetime.utcnow()

    result = await virtual_tryons.update_one(
        {
            "guest_id": guest_id,
            "session_id": session_id
        },
        {
            "$set": {
                "selected_frame": {
                    "frame_id": frame_id,
                    "frame_name": frame_name,
                    "dimensions": frame_dims,
                    "fitting_height": fitting_height,
                    "frame_image_url": frame_image_url,
                    "selected_at": now
                },
                "status.frame_selected": True,
                "updated_at": now
            }
        }
    )

    return result.matched_count > 0

async def get_face_height(
    guest_id: str,
    session_id: str
):
    doc = await virtual_tryons.find_one(
        {
            "guest_id": guest_id,
            "session_id": session_id,
            "status.measurements_done": True
        },
        {
            "_id": 0,
            "measurements.mm.face_height": 1
        }
    )

    if not doc:
        return None

    return doc["measurements"]["mm"]["face_height"]
