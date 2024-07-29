import json

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi.responses import JSONResponse
from FaceNet_Componenet.FaceNet_Utils import process_images, save_embeddings_to_db, get_reference_embeddings, \
    calculate_similarity, logger
from Utils.Log_level import LogLevel, set_log_level
from config.config import FACENET_SERVER_PORT
from Utils.db import embedding_collection, check_mongo


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")
    logger.info("Application stopped.")


app = FastAPI(
    lifespan=lifespan,
    title="Face Comparison API",
    description="This API allows you to set a reference image and compare it with uploaded images to calculate similarity percentages using FaceNet."
)


@app.post("/set_logging_level/", description="Set the logging level dynamically.")
async def set_logging_level(request: LogLevel):
    try:
        set_log_level(request.name,logger)
        return JSONResponse(status_code=200, content={"message": f"Logging level set to {request.name}"})
    except Exception as e:
        logger.error(f"Error setting logging level: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/set_reference_image/", description="Set the reference images for face comparison.")
async def set_reference_image(uuid: str, files: List[UploadFile] = File(...), user_details: str = Form(...)):
    try:
        # Parse user details
        user_details_dict = json.loads(user_details)

        embeddings = await process_images(files)
        if not embeddings:
            return JSONResponse(status_code=400,
                                content={"error": "Failed to calculate any embeddings from the provided images"})

        await save_embeddings_to_db(uuid, embeddings, user_details_dict)
        return JSONResponse(status_code=200, content={
            "message": "Reference images set, embeddings, and average embedding calculated successfully",
            "num_embeddings": len(embeddings)
        })
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid user_details format. It should be a valid JSON string."}, status_code=400)
    except Exception as e:
        logger.error(f"Error setting reference images: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/compare/",
          description="Compare an uploaded image with the reference image and return the similarity percentage.")
async def compare_faces_endpoint(uuid: str, request: Request):
    try:
        record = await get_reference_embeddings(uuid)
        if not record:
            return JSONResponse(status_code=400, content={
                "detail": "Reference embeddings not set. Please use /set_reference_image first."})

        detected_image_base64 = (await request.json()).get("image_base_64")
        similarity_percentage = await calculate_similarity(record, detected_image_base64)
        return JSONResponse(status_code=200, content={"similarity_percentage": similarity_percentage})
    except Exception as e:
        logger.error(f"Error in compare_faces_endpoint: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/health/", description="Health check endpoint to verify that the application is running.")
async def health_check():
    try:
        if check_mongo():
            logger.info("Health check successful.")
            return JSONResponse(content={"status": "healthy"}, status_code=200)
        else:
            logger.warning("MongoDB is not ready.")
            return JSONResponse(content={"status": "unhealthy", "error": "MongoDB is not ready."}, status_code=503)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(content={"status": "unhealthy", "error": str(e)}, status_code=500)




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=FACENET_SERVER_PORT)
