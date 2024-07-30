import logging
import os
from typing import Dict, Any
import ffmpeg
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import requests

from Utils.db import detected_frames_collection, embedding_collection
from Yolo_Componenet.YoloV8Detector import YoloV8Detector
from config.config import FACENET_SERVER_URL, MONGODB_URL
from motor.motor_asyncio import AsyncIOMotorClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = YoloV8Detector("../yolov8l.pt")
face_comparison_server_url = FACENET_SERVER_URL + "/compare/"
client = AsyncIOMotorClient(MONGODB_URL)


async def insert_detected_frames_separately(uuid: str, running_id: str, detected_frames: Dict[str, Any]):
    for frame_index, frame_data in detected_frames.items():
        frame_document = {
            "uuid": uuid,
            "running_id": running_id,
            "frame_index": frame_index,
            "frame_data": frame_data,
            "embedded": False,
        }
        await detected_frames_collection.insert_one(frame_document)


async def process_and_annotate_video(video_path: str, similarity_threshold: float, uuid: str, running_id: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")

    output_path = video_path + "_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for MPEG-4
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if not out.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    frame_index = 0
    detected_frames: Dict[str, Any] = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_obj = detector.predict(frame, frame_index=frame_index)

        await annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid)
        out.write(frame)

    cap.release()
    out.release()

    # Save detected frames to MongoDB separately
    await insert_detected_frames_separately(uuid, running_id, detected_frames)

    # Re-encode the annotated video
    reencoded_output_path = video_path + "_reencoded.mp4"
    reencode_video(output_path, reencoded_output_path)

    if not os.path.exists(reencoded_output_path):
        raise HTTPException(status_code=500, detail="Re-encoded video file not found after processing")

    return reencoded_output_path


async def annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid):
    for detection in frame_obj.detections:
        detected_image_base64 = detection.image_base_64
        response = requests.post(face_comparison_server_url, params={"uuid": uuid},
                                 json={"image_base_64": detected_image_base64})
        if response.status_code == 200:
            similarity = response.json().get("similarity_percentage")
            if similarity is not None and similarity > similarity_threshold:
                x1, y1, x2, y2 = detection.coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{similarity:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                detection.similarity = similarity
                detection.founded = True
                detected_frames[f"frame_{frame_obj.frame_index}"] = {"cropped_image": detection.image_base_64,
                                                                     "similarity": similarity}
                break
            else:
                logger.debug(f"No similarity score or below threshold for detection: {detection.frame_index}")
        else:
            error_message = response.json().get("detail", "Unknown error")
            logger.error(f"Error from face comparison server: {response.status_code} - {error_message}")
            raise HTTPException(status_code=response.status_code,
                                detail=f"Face comparison server error: {error_message}")


def create_streaming_response(file_path: str, filename: str):
    logger.info(f"Creating streaming response for file: {file_path}")
    return StreamingResponse(
        iter_file(file_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


async def iter_file(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, mode="rb") as file_like:
        while chunk := file_like.read(1024):
            yield chunk


async def fetch_detected_frames(uuid: str, running_id: str):
    cursor = detected_frames_collection.find({"uuid": uuid, "running_id": running_id})
    detected_frames = {}
    async for document in cursor:
        frame_index = document["frame_index"]
        frame_data = document["frame_data"]
        detected_frames[frame_index] = frame_data

    extra_details = await embedding_collection.find_one({"uuid": uuid})
    if extra_details:
        detected_frames["user_details"] = extra_details["user_details"]

    return detected_frames


def reencode_video(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', strict='-2').global_args(
            '-loglevel', 'quiet', '-hide_banner').run()
        logger.info("Video re-encoded successfully!")
    except ffmpeg.Error as e:
        logger.error(f"Error occurred during re-encoding: {e.stderr}")
