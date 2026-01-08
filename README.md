# adaptive-ptz-attendance-system

### Features

1. Enrollment Module: Offline processing of ID-card images via FolderWatcher to generate and store facial embeddings.
2. Real-Time Detection: Uses InsightFace (SCRFD) for fast, landmark-aware face detection.
3. Multi-Face Tracking: Simple IoU-based tracker (SimpleTracker) for stable identity assignment across frames.
4. Quality-Aware Selection: Buffers frames per track and selects the highest-scoring one based on sharpness (60%), frontal pose (20%), and brightness (20%).
5. Face Alignment: 5-point landmark-based affine transformation aligned to ArcFace template, with lighting normalization.
6. Embedding Extraction: ArcFace (InsightFace buffalo_l) for 512D L2-normalized vectors.
7. Recognition & Matching: Cosine similarity via pgvector in PostgreSQL; configurable threshold (0.50).
8. Attendance Logging: Cooldown-based insertion (AttendanceLogger) to avoid duplicates; stores metadata (confidence, track ID, source).
9. PTZ Integration: Axis M1137 Mk II camera support with preset positioning (e.g., Entrance View) and basic controls.
10. Unknown Face Logging: Saves unrecognized crops for manual review.

### System Architecture

##### The pipeline processes video frames as follows:
*Input:* PTZ/Webcam stream → Frame capture.
*Detection:* InsightFace SCRFD → Bounding boxes + 5 keypoints.
*Tracking:* SimpleTracker (IoU matching) → Persistent track IDs.
*Quality Selection:* QualitySelector → Best frame per track (min 5 frames buffered).
*Alignment:* FaceAligner → 112×112 normalized crop.
*Embedding: InsightEmbedder → 512D vector.
*Recognition:* Recognizer → DB query (pgvector cosine) → Match if score ≥ threshold.
*Logging:* AttendanceLogger → Insert with cooldown check.

### Edit config/app_config.json:

---
```yaml
{
   "camera_type": "ptz",
    "ptz": {
        "ip": "<IP-Address-Of-Camera>",
        "username": "<Username-of-camera>",
        "password": "<PasswordOfUser>",
        "entrance_preset": "EntranceView"
    },
    "recognition_threshold": 0.35,
    "cooldown_seconds": 20,
    "iou_threshold": 0.25,
    "max_missed": 15,
    "buffer_size": 5,
    "min_frames": 5
}
---
```


### config/db_config.json:
---
```yaml
{
    "host": "<hostname>",
    "port": "<port-no>",
    "user": "<username>",
    "password": "<user-password>",
    "database": "<web_attendance_system>"
}
```
---


