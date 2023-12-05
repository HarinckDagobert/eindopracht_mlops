# main.py

from fastapi import FastAPI, APIRouter
import tensorflow as tf
import cv2

saved_model_path = './model'

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


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray_frame = cv2.resize(gray_frame, (48, 48))


@router.post("/predict")
def set_interval(data: dict):
    global model
    image = capture_image()

    predictions = model.predict(image)

    return {"Predictions": predictions}


app.include_router(router)
