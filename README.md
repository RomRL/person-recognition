# YOLOv8 Detection and Face Comparison API

This project provides an API for processing video files to detect objects using YOLOv8 and compare detected faces with a reference image using FaceNet. The API is built using FastAPI and integrates MongoDB for data storage.

## Features

- Detect objects in video files using YOLOv8.
- Annotate video frames with detected objects and their similarity scores.
- Store and retrieve detected frames and embeddings from MongoDB.
- Dynamically set logging levels and similarity thresholds.
- Perform face comparison using multiple reference images.

## Requirements

- Python 3.10
- FastAPI
- Uvicorn
- OpenCV
- Requests
- Motor (for MongoDB)
- facenet-pytorch
- PIL
- numpy

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure MongoDB:**
    
    Ensure you have MongoDB running locally or provide a connection URL in the `.env` file:

    ```python
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://root:example@localhost:27017/?authMechanism=DEFAULT")
    ```

5. **Run the Application:**

    ```bash
    uvicorn Yolo_Server_Main:app --reload
    uvicorn FaceNet_Main:app --reload
    ```
   
# Server Files
### [Yolo_Server_Main.py](Yolo_Server_Main.py)
### [FaceNet_Main.py](FaceNet_Main.py)

## Endpoints

### YOLOv8 Detection and Face Comparison API

- **`POST /set_logging_level/`**  
  Set the logging level dynamically.
  - Request: `{"level": "INFO"}`

- **`POST /set_threshold/`**  
  Set the similarity threshold for face comparison.
  - Request: `{"threshold": 0.8}`

- **`POST /set_reference_image/`**  
  Set the reference images for face comparison.
  - Request: `uuid=<UUID>` Upload multiple image files.

- **`POST /detect_and_annotate/`**  
  Annotate the uploaded video file with detected objects and similarity scores.
  - Request: `uuid=<UUID>`, `running_id=<RUNNING_ID>`, Upload a video file and provide similarity threshold.

- **`GET /get_detected_frames/`**  
  Get the detected frames from the last processed video.
  - Request: `uuid=<UUID>`, `running_id=<RUNNING_ID>`

- **`GET /health/`**  
  Health check endpoint to verify that the application is running.
  - Request: None

- **`DELETE /purge_detected_frames/`**  
  Purge the detected frames collection.
  - Request: None

### FaceNet Component

- **`POST /set_logging_level/`**  
  Set the logging level dynamically.
  - Request: `{"level": "INFO"}`

- **`POST /set_reference_image/`**  
  Set the reference images for face comparison.
  - Request: `uuid=<UUID>`, `user_details=<USER_DETAILS>`, Upload multiple image files.

- **`POST /compare/`**  
  Compare an uploaded image with the reference image and return the similarity percentage.
  - Request: `uuid=<UUID>`, `{"image_base_64": "<base64_encoded_image>"}`

- **`GET /health/`**  
  Health check endpoint to verify that the application is running.
  - Request: None

## Configuration

The application can be configured using environment variables. These variables allow you to override default settings for the server ports, folder paths, similarity threshold, and MongoDB URL.

### Environment Variables

- `FACENET_SERVER_PORT`: Port for the FaceNet server (default: `8001`)
- `FACENET_SERVER_URL`: URL for the FaceNet server (default: `http://localhost:8001`)
- `FACENET_FOLDER`: Folder path for FaceNet components (default: `<ROOT_PATH>/FaceNet_Componenet`)
- `YOLO_SERVER_PORT`: Port for the YOLO server (default: `8000`)
- `YOLO_SERVER_URL`: URL for the YOLO server (default: `http://localhost:8000`)
- `YOLO_FOLDER`: Folder path for YOLO components (default: `<ROOT_PATH>/Yolo_Componenet`)
- `SIMILARITY_THRESHOLD`: Similarity threshold for face comparison (default: `30.0`)
- `MONGODB_URL`: MongoDB connection URL (default: `mongodb://root:example@localhost:27017/?authMechanism=DEFAULT`)

### Example `.env` File

You can create a `.env` file in the root directory of your project to override the default configuration:

```dotenv
FACENET_SERVER_PORT=9001
FACENET_SERVER_URL=http://localhost:9001
FACENET_FOLDER=/path/to/facenet_folder
YOLO_SERVER_PORT=9000
YOLO_SERVER_URL=http://localhost:9000
YOLO_FOLDER=/path/to/yolo_folder
SIMILARITY_THRESHOLD=25.0
MONGODB_URL=mongodb://username:password@localhost:27017/?authMechanism=DEFAULT
