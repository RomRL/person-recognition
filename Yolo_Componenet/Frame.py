class Frame:
    """
    A class to represent a video frame with its detections.
    """

    def __init__(self, frame_index):
        self.frame_index = frame_index
        self.detections = []

    def add_detection(self, detection):
        self.detections.append(detection)

    def __str__(self):
        return f"Frame {self.frame_index} with {len(self.detections)} detections"

    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "detections": [detection.to_dict() for detection in self.detections]
        }
