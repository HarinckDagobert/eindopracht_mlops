from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import os
import numpy as np
import base64
import uuid

absolute_path = os.path.dirname(__file__)
relative_path = "model"
saved_model_path = os.path.join(absolute_path, relative_path)

model = tf.keras.models.load_model(saved_model_path)

app = FastAPI()

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"api": "Not found"}},
)


@router.get("/test")
def make_api_call():
    response = "response test"
    return {"API Response": response}


def process_image(content):
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_gray_frame = cv2.resize(gray_frame, (48, 48))
    image = resized_gray_frame.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@router.post("/predict")
async def predict_face(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    image = process_image(contents)

    global model
    predictions = model.predict(image)

    emotion_labels = ["angry", "happy", "sad"]

    result_dict = {
        "Predictions": {label: float(value) for label, value in zip(emotion_labels, predictions.tolist()[0])},
        "EncodedImage": base64.b64encode(contents).decode("utf-8")
    }

    return JSONResponse(content=result_dict)


app.include_router(router)
