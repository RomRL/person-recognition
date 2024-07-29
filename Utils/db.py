from motor.motor_asyncio import AsyncIOMotorClient
from config.config import MONGODB_URL

client = AsyncIOMotorClient(MONGODB_URL)
database = client["Person_Recognition"]
embedding_collection = database.get_collection("embeddings")
detected_frames_collection = database.get_collection("detected_frames")


async def check_mongo():
    try:
        # The ping command is used to check if the connection to MongoDB is up and running
        client.admin.command('ping')
        return True
    except ConnectionError:
        return False


def delete_many_embedding_collection(query={}):
    embedding_collection.delete_many(query)


def delete_many_detected_frames_collection(query={}):
    detected_frames_collection.delete_many(query)
