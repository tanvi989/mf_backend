from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.glasses_detector import router as glasses_router
from app.routes.landmark_detector import router as landmark_router
from app.routes.virtual_tryon import virtual_tryon
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
app.include_router(glasses_router)
app.include_router(landmark_router)
app.include_router(virtual_tryon)  # from virtual_tryon.py
# Root endpoint
@app.get("/")
def root():
    return {"message": "API running"}
