FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source from GitHub repo
# (Spaces clones the Space repo, but we need the FER src/ too)
# We copy src/ directly into the Space
COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/
COPY app.py /app/app.py

ENV FER_ROOT=/app
ENV CHECKPOINT_PATH=/app/checkpoints/best_model.pth

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]