import base64
import logging
from typing import List, Dict, Union
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
from fastapi import UploadFile
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceEmbedding:
    def __init__(self, device: str):
        self.device = device
        self.mtcnn = MTCNN(device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        logger.info("MTCNN and InceptionResnetV1 models loaded successfully\n FaceEmbedding instance created")

    @staticmethod
    def preprocess_image(image_base64: str) -> Image.Image:
        """
        Preprocess the image by decoding the base64 string.
        :param image_base64: Base64 encoded image
        :return: Decoded image
        """
        decoded_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        return decoded_image

    def get_embedding(self, image: Image.Image) -> Union[np.ndarray, None]:
        """
        Get the embedding of the face in the image.
        :param image: Image to process
        :return: Embedding of the face in the image if a face is detected, otherwise None
        """
        try:
            img_cropped = self.mtcnn(image)
            if img_cropped is not None:
                img_cropped = img_cropped.to(self.device)  # Ensure the tensor is on the same device as the model
                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = self.resnet(img_cropped.unsqueeze(0)).detach().cpu().numpy()
                logger.debug("Face detected and embedding calculated successfully")
                return img_embedding
            logger.warning("No faces detected in the image.")
        except Exception as e:
            logger.error(f"Error in get_embedding: {e}")
        return None

    @staticmethod
    def compare_faces_embedding(embedding: np.ndarray, embedding_list: List[np.ndarray]) -> float:
        """
        Compare the input embedding with a list of embeddings and return the maximum similarity percentage.
        :param embedding: Face embedding to compare
        :param embedding_list: List of embeddings to compare against
        :return: Maximum similarity percentage
        """
        # Ensure that both embeddings are 2D arrays
        embedding = embedding.reshape(1, -1)
        embedding_array = np.array(embedding_list).reshape(len(embedding_list), -1)

        # Compute cosine similarities in a vectorized manner
        similarities = cosine_similarity(embedding, embedding_array)

        # Convert to percentages
        similarity_percentages = similarities[0] * 100

        # Log the similarity percentages (optional)
        logger.info(f"Similarity percentages: {similarity_percentages}")

        # Return the maximum similarity percentage
        return np.max(similarity_percentages)


class EmbeddingManager:
    def __init__(self, collection):
        self.collection = collection

    async def save_embeddings_to_db(self, uuid: str, new_embeddings: List[np.ndarray], user_details: dict):
        existing_record = await self.collection.find_one({"uuid": uuid})

        if existing_record:
            update_data = await self.handle_existing_record(existing_record, new_embeddings, user_details)
        else:
            update_data = self.handle_new_record(new_embeddings, user_details)

        await self.collection.update_one(
            {"uuid": uuid},
            {"$set": update_data},
            upsert=True
        )
        logger.info("Reference embeddings and average embedding calculated successfully")

    async def process_images(files: List[UploadFile], face_embedding: FaceEmbedding) -> List[np.ndarray]:
        embeddings = []
        for file in files:
            image_bytes = await file.read()
            reference_image = Image.open(io.BytesIO(image_bytes))
            logger.debug("Reference image loaded successfully")
            embedding = face_embedding.get_embedding(reference_image)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.error(f"Failed to calculate embedding for image: {file.filename}")
        return embeddings

    async def handle_existing_record(self, existing_record: Dict, new_embeddings: List[np.ndarray],
                                     user_details: Dict) -> Dict:
        existing_user_details = self.merge_user_details(existing_record, user_details)
        existing_embeddings = self.convert_to_numpy(existing_record.get("embeddings", []))
        unique_embeddings = self.filter_unique_embeddings(new_embeddings, existing_embeddings)
        embeddings_to_save = existing_embeddings + unique_embeddings
        average_embedding = self.calculate_average_embedding(embeddings_to_save, existing_record)

        return {
            "embeddings": [e.tolist() for e in embeddings_to_save],
            "average_embedding": average_embedding,
            "user_details": existing_user_details
        }

    def handle_new_record(self, new_embeddings: List[np.ndarray], user_details: Dict) -> Dict:
        average_embedding = self.calculate_average_embedding(new_embeddings)

        return {
            "embeddings": [e.tolist() for e in new_embeddings],
            "average_embedding": average_embedding,
            "user_details": user_details
        }

    @staticmethod
    def merge_user_details(existing_record: Dict, new_user_details: Dict) -> Dict:
        existing_user_details = existing_record.get("user_details", {})
        existing_user_details.update(new_user_details)
        return existing_user_details

    @staticmethod
    def convert_to_numpy(embeddings: List[Union[List, np.ndarray]]) -> List[np.ndarray]:
        return [np.array(e) for e in embeddings]

    @staticmethod
    def filter_unique_embeddings(new_embeddings: List[np.ndarray], existing_embeddings: List[np.ndarray]) -> List[
        np.ndarray]:
        return [emb for emb in new_embeddings if
                not any(np.array_equal(emb, existing_emb) for existing_emb in existing_embeddings)]

    @staticmethod
    def calculate_average_embedding(embeddings: List[np.ndarray], existing_record: Dict = None) -> List:
        if embeddings:
            return np.mean(embeddings, axis=0).tolist()
        elif existing_record:
            return existing_record.get("average_embedding", [])
        return []

    async def get_reference_embeddings(self, uuid: str):
        return await self.collection.find_one({"uuid": uuid})

    async def calculate_similarity(self, record, detected_image_base64: str, face_embedding: FaceEmbedding) -> float:
        embeddings = [np.array(e) for e in record["embeddings"]]
        detected_image = face_embedding.preprocess_image(detected_image_base64)
        detected_embedding = face_embedding.get_embedding(detected_image)
        if detected_embedding is not None:
            return face_embedding.compare_faces_embedding(embedding=detected_embedding, embedding_list=embeddings)
        else:
            logger.debug("Face not detected in the image")
            return 0
