# web 이미지 (제안서 §11.2). compose의 platform: linux/arm64로 arm64 이미지를 사용한다.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /srv
COPY requirements-common.txt requirements-docker.txt ./
RUN pip install -r requirements-docker.txt
COPY app ./app
