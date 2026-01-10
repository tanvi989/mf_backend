# import io
# import numpy as np
# from PIL import Image
# import mediapipe as mp

# class IrisLandmarkService:
#     IRIS_DIAMETER_MM = 11.7  # industry standard (safe for kids too)

#     mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=True,
#         refine_landmarks=True,
#         max_num_faces=1
#     )

#     # MediaPipe iris landmarks
#     LEFT_IRIS = [474, 475, 476, 477]
#     RIGHT_IRIS = [469, 470, 471, 472]

#     LEFT_IRIS_CENTER = 468
#     RIGHT_IRIS_CENTER = 473

#     @staticmethod
#     def _dist(p1, p2):
#         return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

#     @staticmethod
#     def _mean(points):
#         xs = [p[0] for p in points]
#         ys = [p[1] for p in points]
#         return (sum(xs)/len(xs), sum(ys)/len(ys))

#     @staticmethod
#     def classify_face_shape(width, height, jaw_width, chin_width):
#         ratio = width / height

#         if ratio > 0.90:
#             return "round"
#         elif 0.85 <= ratio <= 0.90:
#             return "square"
#         elif 0.75 <= ratio < 0.85:
#             return "oval"
#         elif ratio < 0.75:
#             return "rectangle"

#         # fallback using chin vs jaw
#         if chin_width < jaw_width * 0.7:
#             return "heart"

#         return "oval"

#     @staticmethod
#     def detect_landmarks(image_bytes: bytes):
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         img = np.array(image)
#         h, w, _ = img.shape

#         result = IrisLandmarkService.mp_face_mesh.process(img)
#         if not result.multi_face_landmarks:
#             raise Exception("No face detected")

#         face = result.multi_face_landmarks[0]
#         points = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

#         # =========================
#         # IRIS SCALE (MM / PX)
#         # =========================
#         left_iris = [points[i] for i in IrisLandmarkService.LEFT_IRIS]
#         right_iris = [points[i] for i in IrisLandmarkService.RIGHT_IRIS]

#         left_diam_px = IrisLandmarkService._dist(left_iris[0], left_iris[2])
#         right_diam_px = IrisLandmarkService._dist(right_iris[0], right_iris[2])
#         iris_diameter_px = (left_diam_px + right_diam_px) / 2

#         if iris_diameter_px <= 0:
#             raise Exception("Invalid iris detection")

#         mm_per_pixel = IrisLandmarkService.IRIS_DIAMETER_MM / iris_diameter_px

#         # =========================
#         # PD — FIXED (IRIS CENTERS)
#         # =========================
#         left_iris_center = points[IrisLandmarkService.LEFT_IRIS_CENTER]
#         right_iris_center = points[IrisLandmarkService.RIGHT_IRIS_CENTER]

#         pd_px = IrisLandmarkService._dist(left_iris_center, right_iris_center)
#         pd_mm = pd_px * mm_per_pixel

#         # =========================
#         # FACE DIMENSIONS
#         # =========================
#         jaw_left = points[234]
#         jaw_right = points[454]
#         chin = points[152]
#         forehead = points[10]

#         face_width_mm = IrisLandmarkService._dist(jaw_left, jaw_right) * mm_per_pixel
#         face_height_mm = IrisLandmarkService._dist(chin, forehead) * mm_per_pixel

#         jaw_width_mm = face_width_mm
#         chin_width_mm = IrisLandmarkService._dist(points[132], points[361]) * mm_per_pixel

#         # =========================
#         # FACE SHAPE
#         # =========================
#         face_shape = IrisLandmarkService.classify_face_shape(
#             face_width_mm,
#             face_height_mm,
#             jaw_width_mm,
#             chin_width_mm
#         )

#         return {
#             "scale": {
#                 "mm_per_pixel": round(mm_per_pixel, 4),
#                 "iris_diameter_px": round(iris_diameter_px, 2)
#             },
#             "mm": {
#                 "pd": round(pd_mm, 1),
#                 "face_width": round(face_width_mm, 1),
#                 "face_height": round(face_height_mm, 1)
#             },
#             "face_shape": face_shape
#         }


import io
import numpy as np
from PIL import Image
import mediapipe as mp

class IrisLandmarkService:
    # Base adult iris diameter (used as reference)
    BASE_IRIS_MM = 11.7  

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

    # MediaPipe iris landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    @staticmethod
    def _dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    @staticmethod
    def _mean(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    @staticmethod
    def classify_face_shape(width, height, jaw_width, chin_width):
        ratio = width / height
        if ratio > 0.90:
            return "round"
        elif 0.85 <= ratio <= 0.90:
            return "square"
        elif 0.75 <= ratio < 0.85:
            return "oval"
        elif ratio < 0.75:
            return "rectangle"
        # fallback using chin vs jaw
        if chin_width < jaw_width * 0.7:
            return "heart"
        return "oval"

    @staticmethod
    def classify_face_class(face_width_px):
        """Classify small, medium, large faces based on pixel width"""
        if face_width_px < 120:  # small face
            return "small"
        elif 120 <= face_width_px < 160:  # medium face
            return "medium"
        else:  # large face
            return "large"

    @staticmethod
    def detect_landmarks(image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(image)
        h, w, _ = img.shape

        result = IrisLandmarkService.mp_face_mesh.process(img)
        if not result.multi_face_landmarks:
            raise Exception("No face detected")

        face = result.multi_face_landmarks[0]
        points = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

        # =========================
        # FACE WIDTH & HEIGHT (pixels)
        # =========================
        jaw_left = points[234]
        jaw_right = points[454]
        chin = points[152]
        forehead = points[10]

        face_width_px = IrisLandmarkService._dist(jaw_left, jaw_right)
        face_height_px = IrisLandmarkService._dist(chin, forehead)

        # Classify face as small, medium, large
        face_class = IrisLandmarkService.classify_face_class(face_width_px)

        # =========================
        # DYNAMIC IRIS DIAMETER (MM) BASED ON FACE CLASS
        # =========================
        if face_class == "small":
            iris_mm = 10.5
        elif face_class == "medium":
            iris_mm = 11.7
        else:  # large
            iris_mm = 12.5

        # =========================
        # IRIS SCALE (MM / PX)
        # =========================
        left_iris = [points[i] for i in IrisLandmarkService.LEFT_IRIS]
        right_iris = [points[i] for i in IrisLandmarkService.RIGHT_IRIS]

        left_diam_px = IrisLandmarkService._dist(left_iris[0], left_iris[2])
        right_diam_px = IrisLandmarkService._dist(right_iris[0], right_iris[2])
        iris_diameter_px = (left_diam_px + right_diam_px) / 2

        if iris_diameter_px <= 0:
            raise Exception("Invalid iris detection")

        mm_per_pixel = iris_mm / iris_diameter_px

        # =========================
        # PD — center-to-center
        # =========================
        left_iris_center = points[IrisLandmarkService.LEFT_IRIS_CENTER]
        right_iris_center = points[IrisLandmarkService.RIGHT_IRIS_CENTER]
        pd_px = IrisLandmarkService._dist(left_iris_center, right_iris_center)
        pd_mm = pd_px * mm_per_pixel

        # =========================
        # FACE DIMENSIONS (MM)
        # =========================
        face_width_mm = face_width_px * mm_per_pixel
        face_height_mm = face_height_px * mm_per_pixel

        jaw_width_mm = face_width_mm
        chin_width_mm = IrisLandmarkService._dist(points[132], points[361]) * mm_per_pixel

        # =========================
        # FACE SHAPE
        # =========================
        face_shape = IrisLandmarkService.classify_face_shape(
            face_width_mm,
            face_height_mm,
            jaw_width_mm,
            chin_width_mm
        )

        return {
            "scale": {
                "mm_per_pixel": round(mm_per_pixel, 4),
                "iris_diameter_px": round(iris_diameter_px, 2),
                "iris_diameter_mm": iris_mm,
                "face_class": face_class,
                "pixel_ratio": round(iris_mm / iris_diameter_px, 2)
            },
            "mm": {
                "pd": round(pd_mm, 1),
                "face_width": round(face_width_mm, 1),
                "face_height": round(face_height_mm, 1)
            },
            "face_shape": face_shape,
            "debug": {
                "expected_error": "±1–2 mm"
            }
        }
