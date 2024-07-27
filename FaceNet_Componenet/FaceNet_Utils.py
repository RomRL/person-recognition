import base64
import logging
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io

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
    :param image:
    :return: embedding of the face in the image if a face is detected, otherwise None
    """
    try:
        faces, _ = mtcnn.detect(image)
        if faces is not None and len(faces) > 0:
            aligned = mtcnn(image)
            if aligned is not None:
                aligned = aligned.unsqueeze(0)  # Add batch dimension
                embedding = resnet(aligned).detach().numpy()
                logger.debug("Face detected and embedding calculated successfully")
                return embedding
        logger.warning("No faces detected in the image.")
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
    return None


def compare_faces_embedding(embedding, embedding_list):
    """
    Compare the input embedding with a list of embeddings and return the maximum similarity percentage.
    :param embedding:
    :param embedding_list:
    :return: maximum similarity percentage
    """
    similarity_percentages = []
    for emb in embedding_list:
        distance = np.linalg.norm(embedding - emb)
        similarity = np.exp(-distance) * 100  # Convert distance to similarity percentage
        similarity_percentages.append(similarity)
    return max(similarity_percentages)


