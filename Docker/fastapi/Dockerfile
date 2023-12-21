FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /app

ARG MODEL_PATH

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1 \
    && python3 -m pip install -U tensorflow[and-cuda]


COPY ./requirements.txt .

RUN pip install -r requirements.txt 

COPY ./main.py .

COPY $MODEL_PATH ./model

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
