# STEP1 Choosing the base image (Slim version is ~150MB vs Ubuntu's ~700MB)
FROM python:3.11-slim

# STEP2 Preventing interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# STEP3 Setting the working directory
WORKDIR /app

# STEP4 Install Runtime Dependencies (Keep these!)
# We separate "Build" deps (compilers) from "Runtime" deps (libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV Runtime Dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Math optimization libs (Keep these for NumPy/ONNX)
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libprotobuf-dev \
    # C++ Runtime
    libstdc++6 \
    # Build Tools (We need these to install pip packages, but we will remove them later)
    build-essential \
    g++ \
    gcc \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# STEP5 Copy and install Python Dependencies
COPY requirements.txt .

# Install dependencies and then immediately remove the compilers to save space
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    # CLEANUP: Remove the heavy compilers now that packages are installed
    apt-get purge -y --auto-remove build-essential g++ gcc cmake python3-dev

# STEP6 Copy the application code
COPY . .

# STEP7 Create directories
RUN mkdir -p /app/unknown_faces /app/Face_images /app/logs /app/config

# STEP8 copying the insightface models dir
COPY .insightface /root/.insightface

# STEP9 Env variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# STEP10 Run application
CMD ["python3", "main.py"]