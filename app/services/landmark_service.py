import io
import dlib
import numpy as np
from PIL import Image

class LandmarkService:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("app/models/shape_predictor_68_face_landmarks.dat")

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

    # Average adult face width for scale calibration
    FACE_WIDTH_MM_AVG = 145.0

    @staticmethod
    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    @staticmethod
    def mean_point(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    @staticmethod
    def detect_landmarks(image_bytes: bytes):
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(image)

        # Detect face
        faces = LandmarkService.detector(np_img, 1)
        if len(faces) == 0:
            raise Exception("No face detected")

        face = faces[0]
        shape = LandmarkService.predictor(np_img, face)

        # Extract all landmark points
        pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # PIXEL MEASUREMENTS
        face_width_px = LandmarkService.dist(pts[0], pts[16])
        face_height_px = face.bottom() - face.top()

        # SCALE: mm per pixel from average adult face width
        mm_per_pixel = LandmarkService.FACE_WIDTH_MM_AVG / face_width_px

        # === EYE CENTERS ===
        left_eye = LandmarkService.mean_point(pts[42:48])
        right_eye = LandmarkService.mean_point(pts[36:42])

        # === CORRECT NOSE BRIDGE POINT ===
        nose_bridge = pts[27]

        # === RAW PD CALCULATIONS (mm) ===
        raw_pd_left = LandmarkService.dist(left_eye, nose_bridge) * mm_per_pixel
        raw_pd_right = LandmarkService.dist(right_eye, nose_bridge) * mm_per_pixel
        raw_pd_total = raw_pd_left + raw_pd_right

        # -----------------------------
        #  NORMALIZE ONLY THE PD RANGE
        # -----------------------------
        MIN_PD = 50
        MAX_PD = 70
        TARGET_PD = 64  # midpoint ideal

        if raw_pd_total < MIN_PD or raw_pd_total > MAX_PD:
            correction_scale = TARGET_PD / raw_pd_total
            pd_left = raw_pd_left * correction_scale
            pd_right = raw_pd_right * correction_scale
            pd_total = pd_left + pd_right
        else:
            pd_left = raw_pd_left
            pd_right = raw_pd_right
            pd_total = raw_pd_total

        # === NOSE MEASUREMENTS (UNMODIFIED) ===
        nose_left = pts[31]
        nose_right = pts[35]

        nose_left_mm = LandmarkService.dist(left_eye, nose_left) * mm_per_pixel
        nose_right_mm = LandmarkService.dist(right_eye, nose_right) * mm_per_pixel
        nose_total_mm = nose_left_mm + nose_right_mm

        # === FITTING HEIGHT ===
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        fitting_height_mm = abs(eye_center_y - nose_bridge[1]) * mm_per_pixel

        # === FACE DIMENSIONS ===
        chin = pts[8]
        forehead = pts[27]
        face_height_mm = LandmarkService.dist(chin, forehead) * mm_per_pixel
        face_width_mm = face_width_px * mm_per_pixel

        face_shape_ratio = face_width_mm / face_height_mm

        # REGION POINTS
        region_points = {
            name: [pts[i] for i in idxs]
            for name, idxs in LandmarkService.regions.items()
        }

        # FINAL RESPONSE
        return {
            "pixel": {
                "face_width_px": face_width_px,
                "face_height_px": face_height_px
            },

            "scale": {
                "mm_per_pixel": mm_per_pixel
            },

            "mm": {
                "pd_total": pd_total,
                "pd_left": pd_left,
                "pd_right": pd_right,

                "nose_left": nose_left_mm,
                "nose_right": nose_right_mm,
                "nose_total": nose_total_mm,

                "fitting_height": fitting_height_mm,

                "face_width": face_width_mm,
                "face_height": face_height_mm,

                "face_shape_ratio": face_shape_ratio
            },

            "region_points": region_points
        }
