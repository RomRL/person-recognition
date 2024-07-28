import base64
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import io

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load pre-trained Inception ResNet model (FaceNet)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def preprocess_image(image_base64):
    decoded_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    return decoded_image


def get_embedding(image):
    faces, _ = mtcnn.detect(image)
    if faces is not None:
        aligned = mtcnn(image)
        if aligned is not None:
            aligned = aligned.unsqueeze(0)  # Add batch dimension
            embedding = resnet(aligned).detach()
            return embedding
    return None


def compare_faces(embedding1, embedding2):
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    return distance.item()


def calculate_similarity(distance, threshold=1.1):
    # Normalize distance to a similarity percentage
    similarity = max(0, 100 * (1 - distance / threshold))
    return similarity


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")


first_image_path = "ronaldo_face.jpg"
second_image_path = "cristiano-ronaldo.jpg"

first_image_base64 = encode_image_to_base64(first_image_path)
second_image_base64 = encode_image_to_base64(second_image_path)

first_image = preprocess_image(first_image_base64)
second_image = preprocess_image(second_image_base64)

first_embedding = get_embedding(first_image)
second_embedding = get_embedding(second_image)

if first_embedding is not None and second_embedding is not None:
    distance = compare_faces(first_embedding, second_embedding)
    similarity_percentage = calculate_similarity(distance)
    print(f"Distance between faces: {distance:.2f}")
    print(f"Similarity: {similarity_percentage:.2f}%")
else:
    print("One of the embeddings is None, face detection failed.")
