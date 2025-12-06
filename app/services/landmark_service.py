import io
import dlib
import numpy as np
from PIL import Image

class LandmarkService:
    # Load models once (global)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("app/models/shape_predictor_68_face_landmarks.dat")

    # Regions mapping
    regions = {
        'jawline': list(range(0, 17)),
        'right_eyebrow': list(range(17, 22)),
        'left_eyebrow': list(range(22, 27)),
        'nose_bridge': list(range(27, 31)),
        'nose': list(range(31, 36)),
        'right_eye': list(range(36, 42)),
        'left_eye': list(range(42, 48)),
        'outer_lip': list(range(48, 60)),
        'inner_lip': list(range(60, 68)),
    }

    @staticmethod
    def detect_landmarks(image_bytes: bytes):
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(image)

        # Detect faces
        faces = LandmarkService.detector(np_img, 1)
        if len(faces) == 0:
            raise Exception("No face detected")

        face = faces[0]

        # Get 68 landmarks
        shape = LandmarkService.predictor(np_img, face)

        # Convert to (x, y) list
        points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Build region_points
        region_points = {
            name: [points[i] for i in idxs]
            for name, idxs in LandmarkService.regions.items()
        }

        # Face width & height
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        # Final structure
        result = {
            "regions": LandmarkService.regions,
            "region_points": region_points,
            "face_width": face_width,
            "face_height": face_height
        }

        return result
