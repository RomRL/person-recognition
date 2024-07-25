import cv2
from ultralytics import YOLO


class Detection:
    """
    A class to represent a single detection.
    """

    def __init__(self, class_id, coordinates, confidence, original_shape, image_patch, box_data, frame_index):
        self.class_id = class_id
        self.coordinates = coordinates
        self.confidence = confidence
        self.width = coordinates[2] - coordinates[0]
        self.height = coordinates[3] - coordinates[1]
        self.center_x = coordinates[0] + self.width / 2
        self.center_y = coordinates[1] + self.height / 2
        self.relative_coordinates = self.get_relative_coordinates(original_shape)
        self.image_patch = image_patch  # Store the image patch corresponding to the detection
        self.box_data = box_data  # ALL YOLO bounding box data
        self.frame_index = frame_index  # Store the frame index where the detection occurred

    def get_relative_coordinates(self, original_shape):
        """
        Calculate coordinates relative to the original image dimensions.
        """
        original_height, original_width = original_shape
        x1, y1, x2, y2 = self.coordinates
        relative_x1 = x1 / original_width
        relative_y1 = y1 / original_height
        relative_x2 = x2 / original_width
        relative_y2 = y2 / original_height
        return [relative_x1, relative_y1, relative_x2, relative_y2]

    def __str__(self):
        return (f"Object type: {self.class_id}\n"
                f"Coordinates: {self.coordinates}\n"
                f"Relative Coordinates: {self.relative_coordinates}\n"
                f"Width: {self.width}, Height: {self.height}\n"
                f"Center (X, Y): ({self.center_x}, {self.center_y})\n"
                f"Probability: {self.confidence}\n"
                f"Image Patch Shape: {self.image_patch.shape if self.image_patch is not None else 'N/A'}\n"
                f"Box Data: {self.box_data}\n---")


def filter_by_class(detections, target_class_id):
    """
    Filter detections by a specific class.
    """
    return [d for d in detections if d.class_id == target_class_id]


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
            class_id = result.names[box.cls[0].item()]
            coordinates = box.xyxy[0].tolist()
            coordinates = [round(x) for x in coordinates]
            confidence = round(box.conf[0].item(), 2)
            box_data = box  # Store the original YOLO bounding box data

            # Crop the image patch corresponding to the detection
            x1, y1, x2, y2 = coordinates
            image_patch = frame[y1:y2, x1:x2] if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[
                0] else None

            detections.append(
                Detection(class_id, coordinates, confidence, original_shape, image_patch, box_data, frame_index))

        return detections

    def process_video(self, video_path, target_class_id=None) -> list[list[Detection]]:
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
            if target_class_id:
                detections = filter_by_class(detections, target_class_id)

            frame_detections.append(detections)

        cap.release()
        return frame_detections


if __name__ == "__main__":
    detector = YOLOv8Detector("yolov8l.pt")
    video_frames: list[list[Detection]] = detector.process_video("videoplayback.mp4", target_class_id='person')

    for frame in video_frames:
        for detection in frame:
            print(detection)
        print("----")