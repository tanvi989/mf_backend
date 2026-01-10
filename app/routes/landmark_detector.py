from fastapi import APIRouter, UploadFile, File
# from app.services.landmark_service import LandmarkService
from app.services.iris_landmark_service import IrisLandmarkService
from app.services.credit_card_measurement_service import CreditCardMeasurementService

router = APIRouter(
    prefix="/landmarks",
    tags=["Landmark Detection"]
)

@router.post("/detect")
async def detect_landmarks(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        # result = LandmarkService.detect_landmarks(image_bytes)
        result = IrisLandmarkService.detect_landmarks(image_bytes)
        # result = CreditCardMeasurementService.process(image_bytes)
        return {
            "success": True,
            "landmarks": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# @router.post("/detect-iris")
# async def detect_landmarks_iris(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         result = IrisLandmarkService.detect_landmarks(image_bytes)

#         return {
#             "success": True,
#             "data": result
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e)
#         }

@router.post("/credit-card")
async def measure_with_credit_card(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = CreditCardMeasurementService.process(image_bytes)

        return {
            "success": True,
            "landmarks": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }