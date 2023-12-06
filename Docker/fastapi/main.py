from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse, HTMLResponse
import tensorflow as tf
import cv2
import os
import numpy as np
import base64

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


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray_frame = cv2.resize(gray_frame, (48, 48))
    image = resized_gray_frame.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, frame


def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


@router.get("/predict")
def predict_face():
    global model
    image, original_image = capture_image()

    predictions = model.predict(image)

    emotion_labels = ["angry", "happy", "sad"]

    result_dict = {
        "Predictions": {label: value for label, value in zip(emotion_labels, predictions.tolist()[0])},
        "EncodedImage": encode_image(original_image)
    }

    html_content = """
    <html>
    <head>
        <title>Predicted Image</title>
    </head>
    <body>
        <h2>Predictions:</h2>
        <pre>{}</pre>
        <h2>Image:</h2>
        <img src="data:image/jpg;base64,{}" alt="predicted_image" />
    </body>
    </html>
    """.format(result_dict["Predictions"], result_dict["EncodedImage"])

    return HTMLResponse(content=html_content)


app.include_router(router)
