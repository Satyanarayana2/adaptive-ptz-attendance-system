# core/adaptive_manager.py
import os
import cv2
import math
import threading

class AdaptiveManager:
    """
    Manages the Adaptive Gallery Lifecycle using deterministic file overwriting.
    """
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.lock = threading.Lock()
        self.save_dir = self.config.get("save_dir", "adaptive_faces")
        self.learning_enabled = True
        os.makedirs(self.save_dir, exist_ok=True)
    
    def set_learning_mode(self, enabled):
        """Allows SessionController to toggle learning on/off based on PTZ view"""
        self.learning_enabled = enabled

    def _calculate_sharpness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_iod(self, kps):
        left_eye = kps['left_eye']
        right_eye = kps['right_eye']
        return math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])

    def process(self, person_id, crop, kps, embedding, sim_score):
        if not self.config.get("enabled", True) or not self.learning_enabled:
            return

        # 1. Calculate Live Physics internally
        sharpness = self._calculate_sharpness(crop)
        iod = self._calculate_iod(kps)

        # 2. The Hard Gates
        if (sharpness >= self.config["min_sharpness"] and 
            iod >= self.config["min_iod"] and 
            sim_score >= self.config["adaptive_min_threshold"]):
            
            # 3. Ask Database to Compete (Notice we don't save any images yet!)
            db_result = self.db.smart_adaptive_update(
                person_id=person_id,
                new_embedding=embedding,
                quality_score=sharpness,
                max_slots=self.config["max_slots_per_person"]
            )
            
            # 4. Native OS Overwrite (Only fires if DB says we won)
            if db_result and db_result.get("action") in ["INSERT", "UPDATE"]:
                final_path = db_result["image_path"]
                cv2.imwrite(final_path, crop)
                print(f"[ADAPTIVE] Physical image saved/overwritten at: {final_path}")