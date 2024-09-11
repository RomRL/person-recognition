import logging
import multiprocessing
import os
from typing import Dict, Any
import ffmpeg
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from FaceNet_Componenet.FaceNet_Utils import embedding_manager, face_embedding
from Utils.db import detected_frames_collection, embedding_collection
from Yolo_Componenet.YoloV8Detector import YoloV8Detector
from config.config import FACENET_SERVER_URL, MONGODB_URL
from motor.motor_asyncio import AsyncIOMotorClient
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
detector = YoloV8Detector("../yolov8l.pt", logger)
face_comparison_server_url = FACENET_SERVER_URL + "/compare/"
client = AsyncIOMotorClient(MONGODB_URL)


async def insert_detected_frames_separately(uuid: str, running_id: str, detected_frames: Dict[str, Any],
                                            frame_per_second: int = 30):
    """
    Insert detected frames separately into the MongoDB collection.
    """
    for frame_index, frame_data in detected_frames.items():
        frame_document = {
            "uuid": uuid,
            "running_id": running_id,
            "frame_index": frame_index,
            "frame_data": frame_data,
            "embedded": False,
            "frame_per_second": frame_per_second
        }
        await detected_frames_collection.insert_one(frame_document)




# List to store processed frames and their indices
annotated_frames = {}
detections_frames = {}

# Initialize a queue for frames to be annotated
frame_queue = queue.Queue()


def annotate_frame_worker(similarity_threshold, detected_frames, uuid, refrence_embeddings):
    """
    Worker function to annotate frames with detected faces.
    """
    global annotated_frames
    while True:
        try:
            item = frame_queue.get()
            if item is None:
                break

            frame, frame_obj, frame_index = item

            # Annotate the frame
            logger.info(f"Annotating frame {frame_obj.frame_index}")
            annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid, refrence_embeddings,
                           frame_index)

            # Safely store the annotated frame in the shared dictionary
            annotated_frames[frame_index] = frame

        except Exception as e:
            logger.error(f"Error in annotate_frame_worker: {e}")
            frame_queue.task_done()  # Ensure the queue task is marked as done even if there's an error

        finally:
            logger.info(f"Finished processing frame {frame_index}, marking as done")
            frame_queue.task_done()


async def process_and_annotate_video(video_path: str, similarity_threshold: float, uuid: str, running_id: str) -> str:
    """
    Process a video file to detect objects using YOLOv8 and annotate the video with the detected faces.
    """
    global annotated_frames  # Make sure the global dictionary is accessible
    annotated_frames = {}
    cap = cv2.VideoCapture(video_path)
    frame_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")
    print_to_log_video_parameters(cap)
    refrence_embeddings = await embedding_manager.get_reference_embeddings(uuid)
    output_path = video_path.replace(".mp4", "_annotated.mp4")
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
    num_annotation_threads = multiprocessing.cpu_count()
    threads = []
    for i in range(num_annotation_threads):
        t = threading.Thread(target=annotate_frame_worker,
                             args=(similarity_threshold, detected_frames, uuid, refrence_embeddings))
        t.start()
        threads.append(t)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1


        # Queue even frames for annotation processing
        if frame_index % 2 == 0:
            # Detect faces
            frame_obj = detector.predict(frame, frame_index=frame_index)

            # Queue the frame for annotation
            frame_queue.put((frame, frame_obj, frame_index))

            logger.info(f"Processing frame {frame_index}/{total_frames}")
        else:
            # Directly add odd frames to the annotated_frames dictionary
            annotated_frames[frame_index] = frame

    # Wait for all frames to be processed
    frame_queue.join()
    logger.info("All frames processed")

    # Stop the worker threads
    for _ in range(num_annotation_threads):
        frame_queue.put(None)
    for t in threads:
        t.join()

    # Write the frames to output video
    for index in range(total_frames):
        frame = annotated_frames.get(index)
        if frame is not None:
            if index % 2 == 1:
                check_and_annotate(index, frame)
            out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Video processing complete , output file saved at {output_path}")
    # Save detected frames to MongoDB separately
    await insert_detected_frames_separately(uuid=uuid, running_id=running_id, detected_frames=detected_frames,
                                            frame_per_second=frame_per_second)



    # Re-encode the annotated video
    reencoded_output_path = video_path.replace(".mp4", "_annotated_reencoded.mp4")

    reencode_video(output_path, reencoded_output_path)

    if not os.path.exists(reencoded_output_path):
        raise HTTPException(status_code=500, detail="Re-encoded video file not found after processing")

    return reencoded_output_path


def check_and_annotate(frame_index, frame):
    """
    Check if the detections in the previous and next frames are similar and annotate the current frame.
    """
    diff_margin = 100
    # check the before and after frame detections and if their cordinates are similar add fiction annotation
    if frame_index - 1 in detections_frames and frame_index + 1 in detections_frames:
        # get the cordinates of the detections
        detection1 = detections_frames[frame_index - 1]
        detection2 = detections_frames[frame_index + 1]
        # check if the cordinates are similar by a margin of error
        if abs(detection1[0][0] - detection2[0][0]) < diff_margin and abs(
                detection1[0][1] - detection2[0][1]) < diff_margin:
            # add the cordinates to the frame
            x1, y1, x2, y2 = detection1[0]
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Draw bounding box in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Position text above the bounding box and ensure it fits within the frame
            text = f"{detection1[1]:.2f}%"
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
            return True



def wrapper(data):
    """
    Wrapper function to calculate similarity between embeddings.
    """
    return embedding_manager.calculate_similarity(
        data[0],
        data[1]
    )


def annotate_frame(frame, frame_obj, similarity_threshold, detected_frames, uuid, refrence_embeddings, frame_index):
    """
    Annotate a frame with detected faces.
    """
    logger.info(f"Found in frame {frame_obj.frame_index}: {len(frame_obj.detections)} detections")
    datas = [(refrence_embeddings, detection.image_base_64) for detection in frame_obj.detections]
    similarities = [wrapper(data) for data in datas]

    for detection, similarity in zip(frame_obj.detections, similarities):

        if similarity is not None and similarity > similarity_threshold:
            logger.info(f"Similarity score: {similarity:.2f}% for detection: {detection.frame_index}, Accepted")
            x1, y1, x2, y2 = detection.coordinates
            detections_frames[frame_index] = (detection.coordinates, similarity)

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
    """
    Calculate similarity between detected image and reference embeddings.
    """
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
    frame_per_second = 30
    detected_frames = {}
    async for document in cursor:
        frame_index = document["frame_index"]
        frame_data = document["frame_data"]

        detected_frames[frame_index] = frame_data
    if detected_frames:
        frame_per_second = document["frame_per_second"]
    extra_details = await embedding_collection.find_one({"uuid": uuid})
    if extra_details:
        detected_frames["user_details"] = extra_details["user_details"]
        detected_frames["frame_per_second"] = frame_per_second

    return detected_frames


def reencode_video(input_path, output_path):
    try:
        logger.info(f"Checking if {input_path} exists...")
        if os.path.exists(input_path):
            logger.info(f"{input_path} exists.")
        else:
            logger.error(f"{input_path} does NOT exist.")

        logger.info(f"Checking if {output_path} is accessible...")
        if os.access(output_path, os.R_OK):
            logger.info(f"{output_path} is readable.")
        else:
            logger.error(f"{output_path} is NOT readable.")

        # Ensure the input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return

        logger.info(f"Input file confirmed: {input_path}")
        logger.info("Re-encoding video...")

        # Run ffmpeg command
        process = (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='libx264', acodec='aac', strict='-2')
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info("Video re-encoded successfully!")

    except Exception as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
        logger.error(f"Error occurred during re-encoding: {e}")
        logger.error(f"An unexpected error occurred during re-encoding: {e}")


def print_to_log_video_parameters(cap):
    logger.info(f"Number of frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    logger.info(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    logger.info(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    logger.info(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
