import os
from google.cloud import storage
from datetime import timedelta
import uuid

BUCKET_NAME = os.getenv("BUCKET_NAME", "multifolks")
BASE_PATH = os.getenv("GCS_BASE_PATH", "vto")

# print("BUCKET_NAME:", BUCKET_NAME)
# print("GCS_BASE_PATH:", BASE_PATH)

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def upload_image_and_get_url(
    file_bytes: bytes,
    guest_id: str,
    session_id: str,
    stage: str,
    ext: str = "jpg",
    expiry_minutes: int = 60
):
    filename = f"{uuid.uuid4()}.{ext}"

    blob_path = (
        f"{BASE_PATH}/{guest_id}/{session_id}/{stage}/{filename}"
    )

    blob = bucket.blob(blob_path)

    blob.upload_from_string(
        file_bytes,
        content_type=f"image/{ext}"
    )

    signed_url = blob.generate_signed_url(
        expiration=timedelta(minutes=expiry_minutes),
        method="GET"
    )

    return {
        "bucket_path": blob_path,
        "signed_url": signed_url
    }
