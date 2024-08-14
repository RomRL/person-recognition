import logging
import os
from typing import Dict, Any
import ffmpeg
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import requests
from FaceNet_Componenet.FaceNet_Utils import embedding_manager, face_embedding
from Utils.db import detected_frames_collection, embedding_collection
from Yolo_Componenet.YoloV8Detector import YoloV8Detector
from config.config import FACENET_SERVER_URL, MONGODB_URL
from motor.motor_asyncio import AsyncIOMotorClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = YoloV8Detector("../yolov8l.pt", logger)
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


import csv
import time

# Initialize a list to store frame processing details
frame_data_list = []

# List to store processed frames and their indices
annotated_frames = []

import threading
import queue
import time

# Initialize a queue for frames to be annotated
frame_queue = queue.Queue()

def annotate_frame_worker(similarity_threshold, detected_frames, uuid, refrence_embeddings):
    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame, frame_obj, frame_index = item

        # Annotate the frame
        logger.info(f"Annotating frame {frame_obj.frame_index}")
        annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid, refrence_embeddings)

        # Store the annotated frame in the list
        annotated_frames.append((frame_index, frame))
        # Mark the task as done
        frame_queue.task_done()


async def process_and_annotate_video(video_path: str, similarity_threshold: float, uuid: str, running_id: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")
    print_to_log_video_parameters(cap)
    refrence_embeddings = await embedding_manager.get_reference_embeddings(uuid)
    output_path = video_path + "_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for MPEG-4
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if not out.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    frame_index = 0
    detected_frames: Dict[str, Any] = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Start a few annotation worker threads
    num_annotation_threads = 4
    threads = []
    for i in range(num_annotation_threads):
        t = threading.Thread(target=annotate_frame_worker, args=(similarity_threshold, detected_frames, uuid, refrence_embeddings))
        t.start()
        threads.append(t)

    start_total = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        start_time = time.time()

        # Detect faces
        detection_start_time = time.time()
        frame_obj = detector.predict(frame, frame_index=frame_index)
        detection_time = time.time() - detection_start_time

        # Queue the frame for annotation
        frame_queue.put((frame, frame_obj, frame_index))

        # Calculate total frame processing time
        total_time = time.time() - start_time

        # Collect frame processing data
        frame_data_list.append({
            "frame_number": frame_index,
            "num_detections": len(frame_obj.detections),
            "detection_time": detection_time,
            "total_time": total_time
        })

        logger.info(f"Processing frame {frame_index}/{total_frames}")


    # Wait for all frames to be processed
    frame_queue.join()

    # Stop the worker threads
    for _ in range(num_annotation_threads):
        frame_queue.put(None)
    for t in threads:
        t.join()

    # Sort frames by their index to ensure correct order
    annotated_frames.sort(key=lambda x: x[0])
     # Write the frames to output video
    for _, frame in annotated_frames:
        out.write(frame)

    logger.info(f"Time annotation: {time.time() - start_total}")
    cap.release()
    out.release()

    # Save detected frames to MongoDB separately
    await insert_detected_frames_separately(uuid, running_id, detected_frames)

    # Write collected data to CSV
    save_frame_data_to_csv(frame_data_list, video_path)

    # Re-encode the annotated video
    reencoded_output_path = video_path + "_reencoded.mp4"
    reencode_video(output_path, reencoded_output_path)

    if not os.path.exists(reencoded_output_path):
        raise HTTPException(status_code=500, detail="Re-encoded video file not found after processing")

    return reencoded_output_path

def save_frame_data_to_csv(frame_data_list, video_path):
    csv_file_path =  "frame_data.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["frame_number", "num_detections", "detection_time", "avg_similarity_time", "total_time"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for frame_data in frame_data_list:
            writer.writerow(frame_data)

    logger.info(f"Frame data saved to {csv_file_path}")

def wrapper(data):
    return embedding_manager.calculate_similarity(
        data[0],
        data[1]
    )


def annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid, refrence_embeddings):
    #from process_pool import process_pool
    logger.info(f"Found in frame {frame_obj.frame_index}: {len(frame_obj.detections)} detections")
    datas = [(refrence_embeddings, detection.image_base_64) for detection in frame_obj.detections]
    #similarities = process_pool.map(wrapper, datas)
    similarities = [wrapper(data) for data in datas]

    for detection, similarity in zip(frame_obj.detections, similarities):

        if similarity is not None and similarity > similarity_threshold:
            logger.info(f"Similarity score: {similarity:.2f}% for detection: {detection.frame_index}, Accepted")
            x1, y1, x2, y2 = detection.coordinates

            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Draw bounding box in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Position text above the bounding box and ensure it fits within the frame
            text = f"{similarity:.2f}%"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.8
            font_thickness = 2

            # Calculate text size and position
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

            # Draw background rectangle for text
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 0, 255), cv2.FILLED)

            # Draw text in white
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

            detection.similarity = similarity
            detection.founded = True
            detected_frames[f"frame_{frame_obj.frame_index}"] = {"cropped_image": detection.image_base_64,
                                                                 "similarity": similarity}
            break
        else:
            logger.debug(f"No similarity score or below threshold for detection: {detection.frame_index}")


async def calculate_similarity(uuid, detected_image_base64):
    reference_embeddings = await embedding_manager.get_reference_embeddings(uuid)
    similarity = await embedding_manager.calculate_similarity(
        reference_embeddings,
        detected_image_base64,
        face_embedding
    )
    return similarity


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
        logger.info("Re-encoding video...")
        ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', strict='-2').global_args(
            '-loglevel', 'quiet', '-hide_banner').run()
        logger.info("Video re-encoded successfully!")
    except ffmpeg.Error as e:
        logger.error(f"Error occurred during re-encoding: {e.stderr}")


def print_to_log_video_parameters(cap):
    logger.info(f"Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    logger.info(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    logger.info(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    logger.info(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
