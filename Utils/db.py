from motor.motor_asyncio import AsyncIOMotorClient
from config.config import MONGODB_URL

async_client = AsyncIOMotorClient(MONGODB_URL)
async_database = async_client["Person_Recognition"]
embedding_collection = async_database.get_collection("embeddings")
detected_frames_collection = async_database.get_collection("detected_frames")


async def check_mongo():
    """
    Health check function to verify that the MongoDB connection is up and running.
    """
    try:
        # The ping command is used to check if the connection to MongoDB is up and running
        async_client.admin.command('ping')
        return True
    except ConnectionError:
        return False


def delete_many_embedding_collection(query={}):
    """
    Delete many documents from the embedding collection.
    """
    embedding_collection.delete_many(query)


def delete_many_detected_frames_collection(query={}):
    """
    Delete many documents from the detected frames collection.
    """
    detected_frames_collection.delete_many(query)
