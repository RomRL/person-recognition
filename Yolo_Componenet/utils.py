import logging

import cv2
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import requests

from Yolo_Componenet.YoloV8Detector import YoloV8Detector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = YoloV8Detector("../yolov8l.pt")
face_comparison_server_url = "http://127.0.0.1:8001/compare/"


def process_and_annotate_video(video_path: str, similarity_threshold: float) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")

    output_path = video_path + "_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame_obj = detector.predict(frame, frame_index=frame_index)

        annotate_frame(frame, frame_obj, similarity_threshold)
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def annotate_frame(frame, frame_obj, similarity_threshold):
    for detection in frame_obj.detections:
        detected_image_base64 = detection.image_base_64
        response = requests.post(face_comparison_server_url, json={"image_base_64": detected_image_base64})
        if response.status_code == 200:
            similarity = response.json().get("similarity_percentage")
            if similarity is not None and similarity > similarity_threshold:
                x1, y1, x2, y2 = detection.coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{similarity:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                logger.warning(f"No similarity score or below threshold for detection: {detection}")
        else:
            logger.error(f"Error from face comparison server: {response.status_code} - {response.text}")


def create_streaming_response(file_path: str, filename: str):
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Type': 'video/mp4',
    }
    return StreamingResponse(iterfile(), headers=headers)
