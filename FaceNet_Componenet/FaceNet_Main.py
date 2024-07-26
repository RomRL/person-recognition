import base64
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import io
from fastapi import FastAPI, Request, UploadFile, File
import uvicorn
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel

from Utils.Log_level import LogLevel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdRequest(BaseModel):
    threshold: float


def preprocess_image(image_base64):
    decoded_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    return decoded_image


def get_embedding(image):
    try:
        faces, _ = mtcnn.detect(image)
        if faces is not None and len(faces) > 0:
            aligned = mtcnn(image)
            if aligned is not None:
                aligned = aligned.unsqueeze(0)  # Add batch dimension
                embedding = resnet(aligned).detach()
                logger.debug("Face detected and embedding calculated successfully")
                return embedding
        logger.warning("No faces detected in the image.")
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
    return None


def compare_faces(embedding1, embedding2):
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
    return distance.item()


def calculate_similarity(distance, threshold=1.1):
    # Normalize distance to a similarity percentage
    similarity = max(0, 100 * (1 - distance / threshold))
    return similarity


# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load pre-trained Inception ResNet model (FaceNet)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

reference_embedding = None
similarity_threshold = 1.1  # Default threshold value


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")
    logger.info("Application stopped.")


app = FastAPI(
    lifespan=lifespan,
    title="Face Comparison API",
    description="This API allows you to set a reference image and compare it with uploaded images to calculate similarity percentages using FaceNet."
)


@app.post("/set_logging_level/", description="Set the logging level dynamically.")
async def set_logging_level(request: LogLevel):
    level = request.name
    logger.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)
    logger.info(f"Logging level set to {level}")
    return {"message": f"Logging level set to {level}"}


@app.post("/set_threshold/", description="Set the similarity threshold for face comparison.")
async def set_threshold(request: ThresholdRequest):
    global similarity_threshold
    similarity_threshold = request.threshold
    logger.info(f"Similarity threshold set to {similarity_threshold}")
    return {"message": f"Similarity threshold set to {similarity_threshold}"}


@app.post("/set_reference_image/", description="Set the reference image for face comparison.")
async def set_reference_image(file: UploadFile = File(...)):
    global reference_embedding
    try:
        image_bytes = await file.read()
        reference_image = Image.open(io.BytesIO(image_bytes))
        logger.debug("Reference image loaded successfully")
        reference_embedding = get_embedding(reference_image)
        if reference_embedding is not None:
            logger.info("Reference embedding calculated successfully")
            return {"message": "Reference image set successfully"}
        else:
            logger.error("Failed to calculate reference embedding")
            return {"error": "Failed to calculate reference embedding"}
    except Exception as e:
        logger.error(f"Error setting reference image: {e}")
        return {"error": str(e)}


@app.post("/compare/",
          description="Compare an uploaded image with the reference image and return the similarity percentage.")
async def compare_faces_endpoint(request: Request):
    try:
        data = await request.json()
        detected_image_base64 = data.get("image_base_64")
        detected_image = preprocess_image(detected_image_base64)
        detected_embedding = get_embedding(detected_image)

        if detected_embedding is not None and reference_embedding is not None:
            distance = compare_faces(detected_embedding, reference_embedding)
            similarity_percentage = calculate_similarity(distance, threshold=similarity_threshold)
            logger.info(f"Similarity percentage: {similarity_percentage}%")
            return {"similarity_percentage": similarity_percentage}
        else:
            logger.error("Face not detected in one or both images")
            return {"error": "Face not detected in one or both images"}
    except Exception as e:
        logger.error(f"Error in compare_faces_endpoint: {e}")
        return {"error": str(e)}


@app.get("/health/", description="Health check endpoint to verify that the application is running.")
async def health_check():
    try:
        logger.info("Health check successful.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
