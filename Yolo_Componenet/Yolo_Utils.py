import logging
import os
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import requests

from Utils.db import detected_frames_collection ,embedding_collection
from Yolo_Componenet.YoloV8Detector import YoloV8Detector
from config.config import FACENET_SERVER_URL, MONGODB_URL
from motor.motor_asyncio import AsyncIOMotorClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = YoloV8Detector("../yolov8l.pt")
face_comparison_server_url = os.path.join(FACENET_SERVER_URL, "compare/")
client = AsyncIOMotorClient(MONGODB_URL)


async def process_and_annotate_video(video_path: str, similarity_threshold: float, uuid: str,running_id:str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")

    output_path = video_path + "_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_index = 0
    detected_frames = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_obj = detector.predict(frame, frame_index=frame_index)

        await annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid,running_id)
        out.write(frame)

    cap.release()
    out.release()

    # Save detected frames to MongoDB
    await detected_frames_collection.update_one(
        {"uuid": uuid},
        {"running_id":running_id},
        {"$set": {"detected_frames": detected_frames}},
        upsert=True
    )

    return output_path


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
                logger.warning(f"No similarity score or below threshold for detection: {detection}")
        else:
            error_message = response.json().get("detail", "Unknown error")
            logger.error(f"Error from face comparison server: {response.status_code} - {error_message}")
            raise HTTPException(status_code=response.status_code,
                                detail=f"Face comparison server error: {error_message}")


def create_streaming_response(file_path: str, filename: str):
    return StreamingResponse(
        iterfile(file_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


async def iterfile(file_path: str):
    with open(file_path, mode="rb") as file_like:
        while True:
            chunk = file_like.read(1024)
            if not chunk:
                break
            yield chunk


async def fetch_detected_frames(uuid: str,running_id:str):
    detected_frames = await detected_frames_collection.find_one({"uuid": uuid},{"running_id":running_id})
    extra_details = await embedding_collection.find_one({"uuid": uuid})
    return {**detected_frames, **extra_details}