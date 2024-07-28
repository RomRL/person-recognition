import base64
import logging
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity

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

# def compare_faces_embedding(embedding, embedding_list):
#     """
#     Compare the input embedding with a list of embeddings and return the maximum similarity percentage.
#     :param embedding:
#     :param embedding_list:
#     :return: maximum similarity percentage
#     """
#     similarity_percentages = []
#     for emb in embedding_list:
#         distance = np.linalg.norm(embedding - emb)
#         similarity = np.exp(-distance) * 100  # Convert distance to similarity percentage
#         similarity_percentages.append(similarity)
#     return max(similarity_percentages)
#


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

        # Transform cosine similarity to range [0, 1]
        similarity = (similarity + 1) / 2  # This transforms [-1, 1] to [0, 1]

        # Convert to percentage
        similarity_percentage = similarity * 100
        similarity_percentages.append(similarity_percentage)

    return max(similarity_percentages)

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
