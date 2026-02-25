# Adaptive PTZ Camera Attendance System

An enterprise-grade, fully automated student attendance system that leverages deep learning, mathematical frame-filtering, and dynamic PTZ (Pan-Tilt-Zoom) camera control. 

Designed for real-world college laboratories, this system solves the **"Domain Gap"** problem (ID card photos vs. live CCTV footage) using an **Adaptive Learning Manager**, optimizes CPU performance using a **Compute Economy Architecture**, and searches faces in milliseconds using **HNSW Vector Indexing**.

---

## System Architecture & Core Modules

### 1. Detection & Tracking Module
* **InsightDetector (SCRFD):** Utilizes the highly optimized SCRFD (Single-Stage Crafter for Face Detection) model from the InsightFace `buffalo_l` pack to find bounding boxes and 5 facial keypoints.
* **KalmanTracker:** Uses Kalman filtering and IoU (Intersection over Union) matching to maintain track IDs across frames. It prevents the system from re-evaluating the same student multiple times per second.

### 2. The Quality Selector (Mathematical Pre-Processing)
To prevent the AI from processing "garbage" frames (heavy motion blur, back of heads, computer monitors), the system uses strict mathematical gates before running heavy facial recognition:
* **Resolution Gate:** Instantly drops bounding boxes smaller than `30x35` pixels.
* **Skin Ratio Gate:** Converts the crop to `YCrCb` color space and drops frames with a skin pixel ratio `< 0.30` (filters out monitors and walls).
* **Buffer & Scoring:** Collects the first 5 valid frames per tracked student and scores them based on *Laplacian Sharpness* (`> 300.0`), *Detection Confidence*, *Frontal Angle (Keypoints)*, and *Brightness*. Only the #1 mathematical best frame is sent to the AI.

### 3. Face Alignment & Embedding
* **Face Aligner:** Uses a 106-point landmark system to mathematically warp and align the face crop so the eyes and nose are perfectly centered.
* **InsightEmbedder (ArcFace):** Processes the aligned face through the `w600k_r50` ResNet model to generate a dense 512-dimensional floating-point vector representing the unique geometry of the face.

### 4. Timetable-Aware Recognition
Instead of comparing a face against the entire college database, the system is **Timetable-Aware**:
* It queries the PostgreSQL `class_schedule` table to see which batch is currently supposed to be in the lab.
* It only searches for Cosine Similarity matches within that specific batch (plus STAFF members), effectively eliminating cross-batch false positives and drastically speeding up the query.
* **Recognition Threshold:** Empirically dialed in to `0.42` to capture PTZ edge-cases without triggering false positives.

### 5. Adaptive Learning Manager (Solving the Domain Gap)
Students rarely look like their ID card photos under harsh lab lighting. 
* During the first 30 minutes, if a student matches their ID card with a score between `0.45` and `0.50`, the system learns this new "domain" face.
* It saves up to **3 adaptive slots** per student directly into the database. 
* It uses pose-aware quality replacement logic to overwrite the weakest adaptive slot if a sharper, better angle is found later.
* **Result:** The system builds a "memory" of the student's appearance over the semester, drastically improving recognition rates.

### 6. Attendance Logger & Cache System
* **L1 Tracker Cache:** Once a student is recognized, their Track ID is cached. For the rest of their movement across the room, the heavy ArcFace Embedder is bypassed, saving massive CPU power.
* **Cooldown Mechanism:** A 20-second cooldown prevents database spam if a student walks in and out of the frame multiple times.
* **First Seen vs. Best Seen:** The database logic uses `ON CONFLICT` to preserve the student's *earliest arrival timestamp*, while safely overwriting their daily log picture if a *higher-confidence* crop is captured later in the session.

### 7. Session Controller & Dynamic PTZ Movement
The system runs completely hands-free by syncing with `timetable.json`. The `SessionController` acts as a state machine:
* **ENTRANCE VIEW (First 30 Minutes):** The camera physically locks onto the door using Axis HTTP CGI commands. The L1 Recognition Cache is **disabled** to force the AI to evaluate every high-quality face, feeding fresh data to the Adaptive Manager.
* **SCANNING PHASE (After 30 Minutes):** The camera automatically alternates between Wide Views and Corner Views to catch seated students. The Recognition Cache is **enabled** to save CPU.
* **IDLE:** When the class ends, the camera returns to a resting position and the AI loop pauses.

### 8. Database Extension & Vector Indexing
The backend is powered by PostgreSQL utilizing the **`pgvector`** extension.
* **HNSW Indexing:** The `face_templates` table uses **Hierarchical Navigable Small World (HNSW)** indexing with `vector_cosine_ops`.
* This allows the database to perform high-dimensional approximate nearest neighbor (ANN) searches in milliseconds, making the system incredibly scalable.

---

## Fine-Tuned Configuration (`app_config.json`)

The system's thresholds have been empirically calculated based on Jupyter Notebook data analysis of over 7,000 real-world lab frames.

```json
{
    "recognition_threshold": 0.42,
    "cooldown_seconds": 20,
    "iou_threshold": 0.25,
    "max_missed": 10,
    "buffer_size": 5,
    "min_frames": 5,
    "adaptive_gallery": {
        "enabled": true,
        "max_slots_per_person": 3,
        "anchor_min_threshold": 0.45,
        "adaptive_min_threshold": 0.50,
        "min_iod": 18.0,
        "min_sharpness": 300.0,
        "save_dir": "adaptive_faces"
    }
}
```

## **Web Interfaces (FastAPI + Jinja2)**

The system features a lightweight web dashboard running on a background thread ([http://localhost:5000](http://localhost:5000)):

**index.html:** The landing page to navigate the application.

**stream.html:** Serves the live multipart/x-mixed-replace JPEG stream. Features a Live HUD (Heads-Up Display) overlaid on the video using OpenCV, displaying the Current Phase, Active Track Count, and Cache Status.

**results.html:** Displays end-of-session Cache Hit-Rate percentages, processing stats, and allows for a graceful multithreading shutdown (`os._exit(0)`).

---

## **Production Deployment (Offline Docker)**

The system is deployed using a "Baked-In" Docker architecture. The ~1.5GB InsightFace AI models (.onnx) are copied directly into the image during the build process, guaranteeing 0 seconds of download time and a lightning-fast, fully offline boot in the lab.

### **1. Build the Blueprint (On your laptop)**

Ensure your `.insightface` folder is in the root directory alongside the Dockerfile, then run:

```bash
docker build -t adaptive_ptz_attendance .
```

### **2. Export to USB**

Compress the built image into a single `.tar` file to physically carry to the lab computer:

```bash
docker save -o adaptive_ptz_attendance.tar adaptive_ptz_attendance
```

### **3. Run in the Lab (Offline)**

Load the image on the lab computer:

```bash
docker load -i adaptive_ptz_attendance.tar
```

Start the orchestrator (which boots the PostgreSQL vector database and the AI App on the host network via `docker-compose.yml`):

```bash
docker compose up -d
```

> **Note:** Ensure the lab computer's OS Power Settings are set to "Never Sleep" to prevent hardware network/camera lockups during long 3-hour lab sessions.
