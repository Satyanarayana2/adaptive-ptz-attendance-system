# STEP1 Choosing the base image
FROM ubuntu:22.04

# STEP2 Preventing interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# INSTALLING system dependencies
RUN apt-get update && apt-get install -y \
    # Python
    python3.11 \
    python3-pip \
    python3-dev \
    # C++ Build Tools (for ONNX Runtime and InsightFace)
    build-essential \
    g++ \
    gcc \
    cmake \
    # OpenCV and Computer Vision libraries
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Math and Performance libraries (CPU optimization)
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    # ONNX Runtime dependencies
    libprotobuf-dev \
    protobuf-compiler \
    # Additional C++ runtime libraries
    libstdc++6 \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# STEP3 Setting the working directory
WORKDIR /app

# STEP4 Copy and install Python Dependencies
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# STEP5 Copy the application code
COPY . .

# STEP6 Creating necessary direcotires
RUN mkdir -p /app/unknown_faces /app/Face_images /app/logs /app/config
# This ensures the folders for "outside files" exist inside the container before the application tries to write to them. You may need to adjust 
# permissions or ownership if your application writes to these directories as a non-root user. For now, this creates the necessary structure for the 
# application to run without errors related to missing directories. 
# should add or change accordingly later looking at the code structure

# STEP7 Pre-download InsightFace models during build
RUN python3 << 'EOF'
import sys
print("[BUILD] Downloading InsightFace models...")

try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(allowed_modules=['detection', 'landmark'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[BUILD] FaceAnalysis models downloaded")
except Exception as e:
    print(f"[BUILD] Warning downloading FaceAnalysis: {e}", file=sys.stderr)

try:
    from insightface.model_zoo import model_zoo
    model = model_zoo.get_model('buffalo_l')
    model.prepare(ctx_id=-1)
    print("[BUILD] Buffalo_l embedding model downloaded")
except Exception as e:
    print(f"[BUILD] Warning downloading embedding model: {e}", file=sys.stderr)

print("[BUILD] Model pre-download complete!")
EOF

# STEP8  setting env variables for the application
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
# CPU thread optimization for ONNX Runtime
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Step 8 run application
CMD ["python3", "main.py"]


