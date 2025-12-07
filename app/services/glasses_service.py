from app.glasses.detector import detector
from PIL import Image
import io
import base64
from app.services.glasses_removal import remove_glasses_service

class GlassesService:

    @staticmethod
    def detect(image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return detector.predict(image)

    @staticmethod
    def remove_glasses(image_bytes: bytes):
        edited_bytes = remove_glasses_service(image_bytes)
        return base64.b64encode(edited_bytes).decode("utf-8")
