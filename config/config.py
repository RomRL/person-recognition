import os
from os.path import join
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FACENET_SERVER_PORT = os.getenv("FACENET_SERVER_PORT", 8001)
FACENET_SERVER_URL = os.getenv("FACENET_SERVER_URL", f"http://localhost:{FACENET_SERVER_PORT}")
FACENET_FOLDER = os.getenv("FACENET_FOLDER",join(ROOT_PATH, "FaceNet_Componenet"))
YOLO_SERVER_PORT = os.getenv("YOLO_SERVER_PORT", 8000)
OLO_SERVER_URL = os.getenv("YOLO_SERVER_URL", f"http://localhost:{YOLO_SERVER_PORT}")
YOLO_FOLDER = os.getenv("YOLO_FOLDER",join(ROOT_PATH, "Yolo_Componenet"))
SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD", 30.0)




