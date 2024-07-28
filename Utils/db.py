from motor.motor_asyncio import AsyncIOMotorClient
from config.config import MONGODB_URL

client = AsyncIOMotorClient(MONGODB_URL)
database = client["Person_Recognition"]
embedding_collection = database.get_collection("embeddings")
detected_frames_collection = database.get_collection("detected_frames")

def check_mongo():
    try:
        # The ping command is used to check if the connection to MongoDB is up and running
        client.admin.command('ping')
        return True
    except ConnectionError:
        return False