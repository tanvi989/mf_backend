from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.settings import settings
import os

# Optional: only needed for GCP, not Mongo
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    settings.GOOGLE_APPLICATION_CREDENTIALS
)


# ✅ Mongo client (NOT string)
client = AsyncIOMotorClient(settings.MONGODB_URI)

# ✅ Database object
db = client[settings.MONGODB_DB_NAME]

# ✅ Collections
guest_sessions = db["guest_sessions"]
virtual_tryons = db["virtual_tryons"]
