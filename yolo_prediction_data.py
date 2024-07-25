import cv2
from ultralytics import YOLO


class Detection:
    """
    A class to represent a single detection.
    """

    def __init__(self, coordinates, confidence, image_patch, frame_index):
        self.founded = False
        self.coordinates = coordinates
        self.confidence = confidence
        self.width = coordinates[2] - coordinates[0]
        self.height = coordinates[3] - coordinates[1]
        self.image_patch = image_patch  # Store the image patch corresponding to the detection
        self.frame_index = frame_index  # Store the frame index where the detection occurred

    def __str__(self):
        return (f"Object type: {self.class_id}\n"
                f"Coordinates: {self.coordinates}\n"
                f"Relative Coordinates: {self.relative_coordinates}\n"
                f"Width: {self.width}, Height: {self.height}\n"
                f"Center (X, Y): ({self.center_x}, {self.center_y})\n"
                f"Probability: {self.confidence}\n"
                f"Image Patch Shape: {self.image_patch.shape if self.image_patch is not None else 'N/A'}\n"
                f"Box Data: {self.box_data}\n---")



class YOLOv8Detector:
    """
    A class to handle YOLOv8 model loading and predictions.
    """

    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def predict(self, frame, frame_index):
        """
        Predict objects in a given frame.
        """
        original_shape = frame.shape[:2]  # Height, Width
        results = self.model.predict(source=frame)
        result = results[0]
        detections = []

        # Extract detection details
        for box in result.boxes:
            coordinates = [round(x) for x in  box.xyxy[0].tolist()]
            confidence = round(box.conf[0].item(), 2)
            # Crop the image patch corresponding to the detection
            x1, y1, x2, y2 = coordinates
            image_patch = frame[y1:y2, x1:x2] if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[
                0] else None

            detections.append(Detection(coordinates, confidence, image_patch, frame_index=frame_index))

        return detections

    def process_video(self, video_path, target_class_id="person") -> list[list[Detection]]:
        """
        Process a video and return detections for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame_index += 1
            if not ret:
                break

            detections = self.predict(frame, frame_index=frame_index)
            frame_detections.append(detections)

        cap.release()
        return frame_detections


if __name__ == "__main__":
    detector = YOLOv8Detector("yolov8l.pt")
    video_frames: list[list[Detection]] = detector.process_video("videoplayback.mp4")

    for frame in video_frames:
        for detection in frame:
            print(detection)
