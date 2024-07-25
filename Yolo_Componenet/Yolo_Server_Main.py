from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import requests
from Yolo_Componenet.Frame import Frame
from Yolo_Componenet.YoloV8Detector import YoloV8Detector

app = FastAPI()
detector = YoloV8Detector("../yolov8l.pt")
face_comparison_server_url = "http://127.0.0.1:8001/compare/"


@app.post("/detect/")
async def detect_video(file: UploadFile = File(...)):
    # Save the uploaded video file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Process the video and get frames with detections
    video_frames: list[Frame] = detector.process_video(tmp_path)

    # Convert frames with detections to a serializable format
    frames_serializable = [frame.to_dict() for frame in video_frames]

    return JSONResponse(content=frames_serializable)


@app.post("/improved_detect/")
async def detect_video(file: UploadFile = File(...)):
    # Save the uploaded video file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Process the video and get frames with detections
    video_frames: list[Frame] = detector.process_video(tmp_path)

    # Convert frames with detections to a serializable format
    frames_serializable = []
    for frame in video_frames:
        frame_data = frame.to_dict()
        for detection in frame_data['detections']:
            detected_image_base64 = detection['image_base_64']
            # Send image to face comparison server
            response = requests.post(face_comparison_server_url, json={"image_base_64": detected_image_base64})
            if response.status_code == 200:
                similarity = response.json().get("similarity_percentage", None)
                detection['similarity'] = similarity
                if similarity and similarity > 20:  # Assuming a threshold of 90% for a match
                    detection["founded"] = True
                    break
        frames_serializable.append(frame_data)

    return JSONResponse(content=frames_serializable)


@app.post("/whoami/", response_model=dict, response_model_exclude_unset=True,
          description="This endpoint returns the filename of the uploaded file.")
async def whoami(string: str):
    return {"filename": string}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
