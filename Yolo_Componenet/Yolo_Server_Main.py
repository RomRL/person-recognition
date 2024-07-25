from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile

from Yolo_Componenet.Frame import Frame
from Yolo_Componenet.YoloV8Detector import YoloV8Detector

app = FastAPI()
detector = YoloV8Detector("../yolov8l.pt")


@app.post("/detect/")
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


@app.post("/whoami/", response_model=dict, response_model_exclude_unset=True,
          description="This endpoint returns the filename of the uploaded file.")
async def whoami(string: str):
    return {"filename": string}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
