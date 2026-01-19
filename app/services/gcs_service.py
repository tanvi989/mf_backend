import os
import uuid
from datetime import timedelta
from app.utils.settings import settings

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    settings.GOOGLE_APPLICATION_CREDENTIALS
)
from google.cloud import storage

# =========================
# CONFIG
# =========================
BUCKET_NAME = settings.BUCKET_NAME
BASE_PATH = settings.FOLDER_NAME  # use FOLDER_NAME consistently

CONTENT_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp"
}

# =========================
# GCS CLIENT
# =========================
client = storage.Client(project=settings.GCP_PROJECT_ID)

# For Requester Pays bucket, pass user_project
bucket = client.bucket(BUCKET_NAME, user_project=settings.GCP_PROJECT_ID)

# =========================
# UPLOAD FUNCTION
# =========================
def upload_image_and_get_url(
    file_bytes: bytes,
    guest_id: str,
    session_id: str,
    stage: str,
    ext: str = "jpg",
    expiry_minutes: int = 60
):
    # Normalize extension
    ext = ext.lower().replace(".", "")
    if ext not in CONTENT_TYPES:
        raise ValueError(f"Unsupported image type: {ext}")

    filename = f"{uuid.uuid4()}.{ext}"
    blob_path = f"{BASE_PATH}/{guest_id}/{session_id}/{stage}/{filename}"
    
    # Create the blob with user_project automatically
    blob = bucket.blob(blob_path)

    # Upload image
    blob.upload_from_string(
        file_bytes,
        content_type=CONTENT_TYPES[ext]
    )

    # Generate signed URL
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiry_minutes),
        method="GET"
    )

    return {
        "bucket_name": BUCKET_NAME,
        "bucket_path": blob_path,
        "signed_url": signed_url
    }
