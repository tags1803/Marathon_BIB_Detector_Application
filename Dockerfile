FROM python:3.10-slim

# System deps for OpenCV, PaddleOCR, video
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/app/.yolo

# Default command
CMD ["python", "test.py"]
