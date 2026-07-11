from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")

client = AsyncIOMotorClient(MONGODB_URL)
db = client.it_support
tickets_collection = db.tickets