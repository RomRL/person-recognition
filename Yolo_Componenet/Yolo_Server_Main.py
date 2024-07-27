import tempfile
from contextlib import asynccontextmanager
import requests
import logging
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from Utils.Log_level import LogLevel
from Yolo_Componenet.Frame import Frame
from Yolo_Componenet.utils import process_and_annotate_video, create_streaming_response, logger, detector, \
    face_comparison_server_url, detected_frames


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")
    logger.info("Application stopped.")


app = FastAPI(
    lifespan=lifespan,
    title="YOLOv8 Detection and Face Comparison API",
    description="This API allows you to process video files to detect objects using YOLOv8 and compare detected faces with a reference image."
)


class SimilarityThresholdRequest(BaseModel):
    similarity_threshold: Optional[float] = 20.0


@app.post("/set_logging_level/", description="Set the logging level dynamically.")
async def set_logging_level(request: LogLevel):
    level = request.name
    logger.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)
    logger.info(f"Logging level set to {level}")
    return {"message": f"Logging level set to {level}"}


@app.post("/detect/", description="Process the uploaded video file and return detections for each frame.")
async def detect_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process the video and get frames with detections
        video_frames: list[Frame] = detector.process_video(tmp_path)

        # Convert frames with detections to a serializable format
        frames_serializable = [frame.to_dict() for frame in video_frames]

        logger.info("Video processed and detections serialized successfully.")
        return JSONResponse(content=frames_serializable)
    except Exception as e:
        logger.error(f"Error in detect_video endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/improved_detect/",
          description="Process the uploaded video file, compare detections with a reference image, and return enhanced detections with similarity scores.")
async def improved_detect_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process the video and get frames with detections
        video_frames: list[Frame] = detector.process_video(tmp_path)

        # Convert frames with detections to a serializable format
        frames_serializable = []
        for frame in video_frames:
            frame_data = frame.to_dict()
            for detection in frame_data['detections']:
                detected_image_base64 = detection['image_base_64']
                # Send image to face comparison server
                response = requests.post(face_comparison_server_url, json={"image_base_64": detected_image_base64})
                if response.status_code == 200:
                    similarity = response.json().get("similarity_percentage")
                    if similarity is not None:
                        detection['similarity'] = similarity
                        if similarity > 20:  # Assuming a threshold of 20% for a match
                            detection["founded"] = True
                            break
                    else:
                        logger.warning(f"No similarity score returned for detection: {detection}")
                else:
                    logger.error(f"Error from face comparison server: {response.status_code} - {response.text}")
            frames_serializable.append(frame_data)

        logger.info("Video processed, detections serialized, and similarities calculated successfully.")
        logger.debug(f"The Response is {frames_serializable}")
        return JSONResponse(content=frames_serializable)
    except Exception as e:
        logger.error(f"Error in improved_detect_video endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/detect_and_annotate/", response_description="Annotated video file")
async def detect_and_annotate_video(file: UploadFile = File(...), similarity_threshold: float = 20.0):
    try:
        # Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        output_path = process_and_annotate_video(tmp_path, similarity_threshold)
        return create_streaming_response(output_path, "annotated_video.mp4")
    except Exception as e:
        logger.error(f"Error in detect_and_annotate_video endpoint: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/get_detected_frames/", description="Get the detected frames from the last processed video.")
async def get_detected_frames():
    try:
        return {"detected_frames": detected_frames , "status": "success"}
    except Exception as e:
        logger.error(f"Error in get_detected_frames endpoint: {e}")
        return {"error": str(e)}


@app.post("/whoami/", response_model=dict, response_model_exclude_unset=True,
          description="This endpoint returns the filename of the uploaded file.")
async def whoami(string: str):
    try:
        logger.info(f"Received request to whoami endpoint with string: {string}")
        return {"filename": string}
    except Exception as e:
        logger.error(f"Error in whoami endpoint: {e}")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
