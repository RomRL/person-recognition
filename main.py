import json
import tempfile
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
from Utils.Log_level import LogLevel, set_log_level
from Yolo_Componenet.Yolo_Utils import process_and_annotate_video, create_streaming_response, logger as yolo_logger, \
    fetch_detected_frames
from FaceNet_Componenet.FaceNet_Utils import embedding_manager, face_embedding
from config.config import YOLO_SERVER_PORT, SIMILARITY_THRESHOLD
from Utils.db import check_mongo, delete_many_detected_frames_collection
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import UploadFile, Form

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

process_pool = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to log application startup and shutdown.
    """
    logger.info("Starting up...")
    yolo_logger.info("Starting up...")
    yield
    logger.info("Shutting down...")
    yolo_logger.info("Shutting down...")
    logger.info("Application stopped.")
    yolo_logger.info("Application stopped.")


app = FastAPI(
    lifespan=lifespan,
    title="YOLOv8 and Face Comparison API",
    description="This API allows you to process video files to detect objects using YOLOv8 and compare detected faces with a reference image using FaceNet."
)
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SimilarityThresholdRequest(BaseModel):
    """
    Pydantic model for the similarity threshold request.
    """
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD


@app.post("/set_logging_level/", description="Set the logging level dynamically.")
async def set_logging_level(request: LogLevel):
    """
    Set the logging level dynamically.
    """
    try:
        set_log_level(request.name, yolo_logger)
        set_log_level(request.name, logger)
        return JSONResponse(status_code=200, content={"message": f"Logging level set to {request.name}"})
    except Exception as e:
        logger.error(f"Error setting logging level: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/detect_and_annotate/", response_description="Annotated video file")
async def detect_and_annotate_video(uuid: str, running_id: str, file: UploadFile = File(...),
                                    similarity_threshold: float = 20.0):
    """
    Process a video file to detect objects using YOLOv8 and annotate the video with the detected
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            yolo_logger.info(f"Temporary file created at {tmp_path}")

        output_path = await process_and_annotate_video(tmp_path, similarity_threshold, uuid, running_id)
        yolo_logger.info(f"Annotated video file created at {output_path}")

        return create_streaming_response(output_path, f"{uuid}_{running_id}_annotated_video.mp4")
    except Exception as e:
        yolo_logger.error(f"Error in detect_and_annotate_video endpoint:\n {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/set_reference_image/", description="Set the reference images for face comparison.")
async def set_reference_image(
        uuid: str,
        files: List[UploadFile] = File(...),
        user_json: str = Form(...)
):
    """
    Set the reference images for face comparison , calculate the embeddings, and save them to the database.
    Returns the number of embeddings saved.
    """
    try:
        # Process images from files
        file_embeddings = await embedding_manager.process_images(files, face_embedding)

        # Query detected_frames_collection for documents with similarity > 80 and the same uuid
        detected_embeddings = await embedding_manager.process_detected_frames(uuid, face_embedding)

        # Combine file embeddings and detected frame embeddings
        new_embeddings = file_embeddings + detected_embeddings

        # Parse the JSON string into a dictionary
        user_details = json.loads(user_json)

        # Save unique embeddings to embedding_collection along with user details
        if new_embeddings:
            await embedding_manager.save_embeddings_to_db(uuid, new_embeddings, user_details=user_details)

        return JSONResponse(status_code=200, content={
            "message": "Reference images set, embeddings, and average embedding calculated successfully",
            "num_embeddings": len(new_embeddings)
        })
    except Exception as e:
        logger.error(f"Error setting reference images: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/get_detected_frames/", description="Get the detected frames from the last processed video.")
async def get_detected_frames(uuid: str, running_id: str):
    """
    Get the detected frames from running_id.
    """
    try:
        detected_frames = await fetch_detected_frames(uuid, running_id)
        if detected_frames and detected_frames:
            return JSONResponse(content={"detected_frames": detected_frames, "status": "success"}, status_code=200)
        return JSONResponse(content={"status": "error", "message": "No detected frames found for the given UUID."},
                            status_code=404)
    except Exception as e:
        yolo_logger.error(f"Error in get_detected_frames endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/health_facenet/", description="Health check endpoint to verify that the FaceNet application is running.")
async def health_check_facenet():
    """
    Health check endpoint to verify that the FaceNet application is running.
    """
    try:
        if check_mongo():
            logger.info("Health check successful.")
            return JSONResponse(content={"status": "healthy"}, status_code=200)
        else:
            logger.warning("MongoDB is not ready.")
            return JSONResponse(content={"status": "unhealthy", "error": "MongoDB is not ready."}, status_code=503)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)

@app.get("/health_yolo/", description="Health check endpoint to verify that the YOLO application is running.")
async def health_check_yolo():
    """
    Health check endpoint to verify that the YOLO application is running.
    """
    try:
        if await check_mongo():
            yolo_logger.info("Health check successful.")
            return JSONResponse(content={"status": "healthy"}, status_code=200)
        else:
            yolo_logger.warning("MongoDB is not ready.")
            return JSONResponse(content={"status": "unhealthy", "error": "MongoDB is not ready."}, status_code=503)
    except Exception as e:
        yolo_logger.error(f"Health check failed: {e}")
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)


@app.delete("/purge_detected_frames/", description="Purge the detected frames collection.")
async def purge_detected_frames():
    """
    Purge the detected frames collection.
    """
    try:
        delete_many_detected_frames_collection()
        return JSONResponse(content={"message": "Detected frames collection purged successfully."}, status_code=200)
    except Exception as e:
        yolo_logger.error(f"Error purging detected frames collection: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
    """
    Start the FastAPI application.
    """
    uvicorn.run(app, host="0.0.0.0", port=YOLO_SERVER_PORT)  # Adjust the port as needed
