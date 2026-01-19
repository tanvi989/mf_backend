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
    # Reference iris diameters (mm)
    IRIS_MM_SMALL = 10.5
    IRIS_MM_MEDIUM = 11.7
    IRIS_MM_LARGE = 12.5

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

    # Iris landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    @staticmethod
    def _dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def classify_face_class(face_width_px):
        if face_width_px < 120:
            return "small"
        elif face_width_px < 160:
            return "medium"
        return "large"

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
        if chin_width < jaw_width * 0.7:
            return "heart"
        return "oval"

    @staticmethod
    def detect_landmarks(image_bytes: bytes):
        # =========================
        # LOAD IMAGE
        # =========================
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(image)
        h, w, _ = img.shape

        result = IrisLandmarkService.mp_face_mesh.process(img)
        if not result.multi_face_landmarks:
            raise Exception("No face detected")

        face = result.multi_face_landmarks[0]
        points = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

        # =========================
        # FACE SIZE (PX)
        # =========================
        jaw_left = points[234]
        jaw_right = points[454]
        chin = points[152]
        forehead = points[10]

        face_width_px = IrisLandmarkService._dist(jaw_left, jaw_right)
        face_height_px = IrisLandmarkService._dist(chin, forehead)

        face_class = IrisLandmarkService.classify_face_class(face_width_px)

        # =========================
        # IRIS SIZE SELECTION
        # =========================
        if face_class == "small":
            iris_mm = IrisLandmarkService.IRIS_MM_SMALL
        elif face_class == "medium":
            iris_mm = IrisLandmarkService.IRIS_MM_MEDIUM
        else:
            iris_mm = IrisLandmarkService.IRIS_MM_LARGE

        # =========================
        # IRIS SCALE
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
        # IRIS CENTERS
        # =========================
        left_eye = points[IrisLandmarkService.LEFT_IRIS_CENTER]
        right_eye = points[IrisLandmarkService.RIGHT_IRIS_CENTER]

        # =========================
        # TOTAL PD (HORIZONTAL)
        # =========================
        eye_dx_px = abs(left_eye[0] - right_eye[0])
        pd_mm = eye_dx_px * mm_per_pixel

        # =========================
        # CORRECT MONOCULAR PD
        # =========================
        mid_x = (left_eye[0] + right_eye[0]) / 2

        left_pd_mm = abs(mid_x - left_eye[0]) * mm_per_pixel
        right_pd_mm = abs(right_eye[0] - mid_x) * mm_per_pixel

        # =========================
        # FACE DIMENSIONS (MM)
        # =========================
        face_width_mm = face_width_px * mm_per_pixel
        face_height_mm = face_height_px * mm_per_pixel
        face_ratio = face_width_mm / face_height_mm

        # =========================
        # NOSE BRIDGE
        # =========================
        nose_center = points[1]
        nose_left = points[94]
        nose_right = points[331]

        nose_bridge_left_mm = (
            IrisLandmarkService._dist(nose_center, nose_left) * mm_per_pixel
        )
        nose_bridge_right_mm = (
            IrisLandmarkService._dist(nose_center, nose_right) * mm_per_pixel
        )

        # =========================
        # FACE SHAPE
        # =========================
        jaw_width_mm = face_width_mm
        chin_width_mm = (
            IrisLandmarkService._dist(points[132], points[361]) * mm_per_pixel
        )

        face_shape = IrisLandmarkService.classify_face_shape(
            face_width_mm,
            face_height_mm,
            jaw_width_mm,
            chin_width_mm
        )

        # =========================
        # FINAL RESPONSE
        # =========================
        return {
            "scale": {
                "mm_per_pixel": round(mm_per_pixel, 4),
                "iris_diameter_px": round(iris_diameter_px, 2)
            },
            "mm": {
                "pd": round(pd_mm, 1),
                "pd_left": round(left_pd_mm, 1),
                "pd_right": round(right_pd_mm, 1),

                "face_width": round(face_width_mm, 1),
                "face_height": round(face_height_mm, 1),
                "face_ratio": round(face_ratio, 2),

                "nose_bridge_left": round(nose_bridge_left_mm, 1),
                "nose_bridge_right": round(nose_bridge_right_mm, 1)
            },
            "face_shape": face_shape
        }
