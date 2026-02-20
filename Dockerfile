FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV (libgl1 replaces libgl1-mesa-glx in Debian trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY app.py /app/app.py

# Create checkpoints dir
RUN mkdir -p /app/checkpoints

ENV FER_ROOT=/app
ENV CHECKPOINT_PATH=/app/checkpoints/best_model.pth

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]