FROM python:3.10-alpine

WORKDIR /app

ARG MODEL_PATH=${MODEL_PATH}

ENV MODEL_PATH=${MODEL_PATH}

COPY ./requirements.txt .

# RUN apk --no-cache add \
#     build-base \
#     musl-dev \
#     python3-dev \
#     cython \
#     linux-headers

RUN pip install -r requirements.txt 

RUN python3 -m pip install tensorflow

# RUN python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1 \
#     && python3 -m pip install -U tensorflow[and-cuda]

COPY ./main.py .

COPY $MODEL_PATH ./model

EXPOSE 80

CMD ["python", "main.py"]
