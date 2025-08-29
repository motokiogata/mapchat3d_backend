FROM python:3.10-slim-bullseye

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

ENTRYPOINT ["python", "docker-entrypoint.py"]