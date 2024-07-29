import base64
import logging
from typing import List

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io

from fastapi import UploadFile
from sklearn.metrics.pairwise import cosine_similarity

from Utils.db import embedding_collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize MTCNN for face detection
mtcnn = MTCNN()
# Load pre-trained Inception ResNet model (FaceNet)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_image(image_base64):
    """
    Preprocess the image by decoding the base64 string.
    :param image_base64:
    :return:
    """
    decoded_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    return decoded_image


def get_embedding(image):
    """
    Get the embedding of the face in the image.
    :param image: image to process
    :return: embedding of the face in the image if a face is detected, otherwise None
    """
    try:
        img_cropped = mtcnn(image)
        if img_cropped is not None:
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = resnet(img_cropped.unsqueeze(0)).detach().numpy()
            logger.debug("Face detected and embedding calculated successfully")
            return img_embedding
        logger.warning("No faces detected in the image.")
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
    return None


def compare_faces_embedding(embedding, embedding_list):
    """
    Compare the input embedding with a list of embeddings and return the maximum similarity percentage.
    :param embedding: face embedding to compare
    :param embedding_list: list of embeddings to compare against
    :return: maximum similarity percentage
    """
    # Reshape embeddings to 2D arrays if necessary
    embedding = embedding.reshape(1, -1)
    embedding_list = [emb.reshape(1, -1) for emb in embedding_list]

    similarity_percentages = []
    for emb in embedding_list:
        similarity = cosine_similarity(embedding, emb)[0][0]

        # Convert to percentage
        similarity_percentage = similarity * 100
        similarity_percentages.append(similarity_percentage)

    return max(similarity_percentages)


async def process_images(files: List[UploadFile]):
    embeddings = []
    for file in files:
        image_bytes = await file.read()
        reference_image = Image.open(io.BytesIO(image_bytes))
        logger.debug("Reference image loaded successfully")
        embedding = get_embedding(reference_image)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            logger.error(f"Failed to calculate embedding for image: {file.filename}")
    return embeddings


async def save_embeddings_to_db(uuid: str, new_embeddings: List[np.ndarray], user_details: dict):
    existing_record = await embedding_collection.find_one({"uuid": uuid})
    if existing_record:
        # Merge existing user details with new user details
        existing_user_details = existing_record.get("user_details", {})
        existing_user_details.update(user_details)

        # Get the existing embeddings
        existing_embeddings = existing_record.get("embeddings", [])
        # Convert existing embeddings to numpy arrays for comparison
        existing_embeddings = [np.array(e) for e in existing_embeddings]

        # Add only new embeddings that are not already in the existing embeddings
        unique_embeddings = []
        for emb in new_embeddings:
            if not any(np.array_equal(emb, existing_emb) for existing_emb in existing_embeddings):
                unique_embeddings.append(emb)

        # Append the unique embeddings to the existing embeddings
        embeddings_to_save = existing_embeddings + unique_embeddings

        # Calculate the new average embedding
        if embeddings_to_save:
            average_embedding = np.mean(embeddings_to_save, axis=0).tolist()
        else:
            average_embedding = existing_record.get("average_embedding", [])

        await embedding_collection.update_one(
            {"uuid": uuid},
            {"$set": {"embeddings": [e.tolist() for e in embeddings_to_save], "average_embedding": average_embedding, "user_details": existing_user_details}},
            upsert=True
        )
    else:
        # Calculate the average embedding for new embeddings
        if new_embeddings:
            average_embedding = np.mean(new_embeddings, axis=0).tolist()
        else:
            average_embedding = existing_record.get("average_embedding", [])
        await embedding_collection.update_one(
            {"uuid": uuid},
            {"$set": {"embeddings": [e.tolist() for e in new_embeddings], "average_embedding": average_embedding, "user_details": user_details}},
            upsert=True
        )
    logger.info("Reference embeddings and average embedding calculated successfully")


async def get_reference_embeddings(uuid: str):
    return await embedding_collection.find_one({"uuid": uuid})


async def calculate_similarity(record, detected_image_base64):
    embeddings = [np.array(e) for e in record["embeddings"]]
    detected_image = preprocess_image(detected_image_base64)
    detected_embedding = get_embedding(detected_image)
    if detected_embedding is not None:
        return compare_faces_embedding(embedding=detected_embedding, embedding_list=embeddings)
    else:
        logger.debug("Face not detected in the image")
        return 0

# def euclidean_distance(vec1, vec2):
#     """
#     Calculate the Euclidean distance between two vectors.
#     :param vec1: First vector
#     :param vec2: Second vector
#     :return: Euclidean distance
#     """
#
#     return np.linalg.norm(vec1 - vec2)
#
#
# def normalize_distances(distances):
#     """
#     Normalize distances to a 0-1 range.
#     :param distances: List of distances
#     :return: List of normalized distances
#     """
#     max_distance = max(distances)
#     min_distance = min(distances)
#
#     # Handle case where all distances are the same
#     if max_distance == min_distance:
#         return [1.0 for _ in distances]
#
#     normalized = [(max_distance - distance) / (max_distance - min_distance) for distance in distances]
#     return normalized
#
#
# def compare_faces_embedding(embedding, embedding_list):
#     """
#     Compare the input embedding with a list of embeddings and return the maximum similarity percentage.
#     :param embedding: face embedding to compare
#     :param embedding_list: list of embeddings to compare against
#     :return: maximum similarity percentage
#     """
#     distances = [euclidean_distance(embedding, emb) for emb in embedding_list]
#     normalized_similarities = normalize_distances(distances)
#
#     # Convert to percentage
#     similarity_percentages = [similarity * 100 for similarity in normalized_similarities]
#
#     return max(similarity_percentages)
