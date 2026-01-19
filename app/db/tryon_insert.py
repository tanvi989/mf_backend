from datetime import datetime
from app.db.mongo import virtual_tryons
async def insert_on_detection(
    guest_id: str,
    session_id: str,
    glasses_detected: bool,
    confidence: float
):
    now = datetime.utcnow()

    await virtual_tryons.update_one(
        {
            "guest_id": guest_id,
            "session_id": session_id
        },
        {
            "$setOnInsert": {
                "guest_id": guest_id,
                "session_id": session_id,
                "images": {},
                "measurements": {},
                "created_at": now
            },
            "$set": {
                "glasses.detected": glasses_detected,
                "glasses.confidence": confidence,
                "status.inserted": True,
                "status.glasses_removed": False,
                "status.measurements_done": False,
                "updated_at": now
            }
        },
        upsert=True
    )
