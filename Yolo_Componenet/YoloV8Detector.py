import cv2
from ultralytics import YOLO
import torch
from Yolo_Componenet.Frame import Frame
from Yolo_Componenet.Detection import Detection


class YoloV8Detector:
    """
    A class to handle YOLOv8 model loading and predictions.
    """

    def __init__(self, model_path, logger):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.logger = logger
        self._choose_running_device()

    def _choose_running_device(self):
        """
        Choose the appropriate device to run the model on.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.logger.info(
            f'Model running on device: {next(self.model.parameters()).device}')  # Check device of the model again
        if device == "cuda":
            self.logger.info(f"Number of GPUs available: {torch.cuda.device_count()}\n"
                             f"GPU name: {torch.cuda.get_device_name(0)}")

    def predict(self, frame, frame_index):
        """
        Predict objects in a given frame.
        """
        results = self.model.predict(source=frame, classes=0)
        result = results[0]
        frame_obj = Frame(frame_index)

        # Extract detection details
        for box in result.boxes:
            coordinates = [round(x) for x in box.xyxy[0].tolist()]
            confidence = round(box.conf[0].item(), 2)
            # Crop the image patch corresponding to the detection
            x1, y1, x2, y2 = coordinates
            image_patch = frame[y1:y2, x1:x2] if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[
                0] else None

            detection = Detection(coordinates, confidence, image_patch, frame_index=frame_index)
            frame_obj.add_detection(detection)

        return frame_obj

    def process_video(self, video_path) -> list[Frame]:
        """
        Process a video and return frames with detections.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame_index += 1
            if not ret:
                break

            frame_obj = self.predict(frame, frame_index=frame_index)
            frames.append(frame_obj)

        cap.release()
        return frames
