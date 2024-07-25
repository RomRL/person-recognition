import cv2
import numpy as np
import base64
from fastapi import FastAPI, Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize

app = FastAPI()
facenet_model = load_model('path_to_your_facenet_model.h5')

# Hardcoded reference image in base64
reference_image_base64 = "your_reference_image_base64_string"
reference_image = cv2.imdecode(np.frombuffer(base64.b64decode(reference_image_base64), np.uint8), cv2.IMREAD_COLOR)
reference_embedding = get_embedding(facenet_model, reference_image)

def preprocess_image(img):
    img = cv2.resize(img, (160, 160))  # Resize to FaceNet input size
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(model, img):
    processed_img = preprocess_image(img)
    embedding = model.predict(processed_img)
    return normalize(embedding)

def compare_faces(embedding1, embedding2):
    similarity = np.linalg.norm(embedding1 - embedding2)
    return similarity

@app.post("/compare/")
async def compare_faces_endpoint(request: Request):
    data = await request.json()
    detected_image_base64 = data.get("image_base_64")
    detected_image = cv2.imdecode(np.frombuffer(base64.b64decode(detected_image_base64), np.uint8), cv2.IMREAD_COLOR)
    detected_embedding = get_embedding(facenet_model, detected_image)
    similarity = compare_faces(detected_embedding, reference_embedding)
    return {"similarity": similarity}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
