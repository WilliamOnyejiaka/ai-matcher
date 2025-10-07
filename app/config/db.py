import motor.motor_asyncio
from app.config.env import DATABASE_URL

mongo_client = motor.motor_asyncio.AsyncIOMotorClient(DATABASE_URL)
db = mongo_client['blumdate_db']
