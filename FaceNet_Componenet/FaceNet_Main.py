import base64
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import io
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load pre-trained Inception ResNet model (FaceNet)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Hardcoded reference image in base64
reference_image_base64 = "your_reference_image_base64_string"

# Decode the reference image
reference_image = Image.open(io.BytesIO(base64.b64decode(reference_image_base64)))
reference_embedding = None

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

@app.on_event("startup")
async def startup_event():
    global reference_embedding
    reference_embedding = get_embedding(reference_image)

@app.post("/compare/")
async def compare_faces_endpoint(request: Request):
    data = await request.json()
    detected_image_base64 = data.get("image_base_64")
    detected_image = preprocess_image(detected_image_base64)
    detected_embedding = get_embedding(detected_image)

    if detected_embedding is not None and reference_embedding is not None:
        distance = compare_faces(detected_embedding, reference_embedding)
        similarity_percentage = calculate_similarity(distance)
        return {"similarity_percentage": similarity_percentage}
    else:
        return {"error": "Face not detected in one or both images"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
