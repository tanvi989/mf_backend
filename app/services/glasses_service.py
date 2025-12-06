from glasses.detector import detector
from PIL import Image
import io

class GlassesService:

    @staticmethod
    def detect(image_bytes: bytes):
        # Convert bytes → PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Use existing detector from detector.py
        result = detector.predict(image)

        return result
