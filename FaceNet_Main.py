import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import List
from FaceNet_Componenet.FaceNet_Utils import logger, preprocess_image, get_embedding, compare_faces_embedding
from Utils.Log_level import LogLevel
from config.config import FACENET_SERVER_PORT

embeddings = None


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


@app.post("/set_reference_image/", description="Set the reference images for face comparison.")
async def set_reference_image(files: List[UploadFile] = File(...)):
    global embeddings
    try:
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

        if embeddings:
            # Calculate the average embedding
            average_embedding = np.mean(embeddings, axis=0)
            embeddings.append(average_embedding)
            logger.info("Reference embeddings and average embedding calculated successfully")
            return {"message": "Reference images set, embeddings, and average embedding calculated successfully", "num_embeddings": len(embeddings)}
        else:
            logger.error("Failed to calculate any embeddings from the provided images")
            return {"error": "Failed to calculate any embeddings from the provided images"}
    except Exception as e:
        logger.error(f"Error setting reference images: {e}")
        return {"error": str(e)}


@app.post("/compare/",
          description="Compare an uploaded image with the reference image and return the similarity percentage.")
async def compare_faces_endpoint(request: Request):
    if embeddings is None:
        logger.error("Reference embeddings not set. Please use /set_reference_image first.")
        raise HTTPException(status_code=400,
                            detail="Reference embeddings not set. Please use /set_reference_image first.")
    try:
        data = await request.json()
        detected_image_base64 = data.get("image_base_64")
        detected_image = preprocess_image(detected_image_base64)
        detected_embedding = get_embedding(detected_image)

        if detected_embedding is not None:
            similarity_percentage = compare_faces_embedding(embedding=detected_embedding, embedding_list=embeddings)
            logger.info(f"Similarity percentage: {similarity_percentage}%")
            return {"similarity_percentage": similarity_percentage}
        else:
            logger.debug("Face not detected in one or both images")
            return {"similarity_percentage": 0}
    except Exception as e:
        logger.error(f"Error in compare_faces_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/", description="Health check endpoint to verify that the application is running.")
async def health_check():
    try:
        logger.info("Health check successful.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=FACENET_SERVER_PORT)
