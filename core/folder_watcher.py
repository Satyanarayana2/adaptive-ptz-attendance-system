import os
import cv2

from utils.db import Database
from utils.detectors.insight_detector import InsightDetector
from utils.embeddings.insight_embedder import InsightEmbedder
from utils.face_alignment import FaceAligner

class FolderWatcher:
    """
    Offline face enrollment service.
    Reads images from Batch-Section subfolders (Students) and the root folder (Staff).
    Detects faces, aligns them, generates embeddings, and stores them in DB under the correct class.
    """

    def __init__(self, image_dir="Face_images"):
        self.image_dir = image_dir
        self.db = Database()
        self.detector = InsightDetector()
        self.detector.prepare()
        self.aligner = FaceAligner()
        self.embedder = InsightEmbedder()
        self.embedder.prepare()

    def _parse_filename(self, filename):
        """Expected format: AM.SC.P2ARI24009_SATYA.jpg"""
        name_part = os.path.splitext(filename)[0]
        if "_" not in name_part:
            return None, None
        parts = name_part.split("_", 1)
        return parts[0], parts[1]

    def _enroll_single_image(self, image_path, class_id, batch, section):
        """Helper function to process a single image and save it to the DB."""
        filename = os.path.basename(image_path)
        image_ref = os.path.relpath(image_path)

        # Step 2: Check duplicate image
        if self.db.image_ref_exists(image_ref):
            print(f"[SKIP] Already enrolled: {filename}")
            return

        # Step 3: Parse filename
        roll_number, name = self._parse_filename(filename)
        if not roll_number or not name:
            print(f"[SKIP] Invalid filename format: {filename}")
            return

        # Step 4: Process image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[SKIP] Cannot read image: {filename}")
            return

        faces = self.detector.detect(img)
        if len(faces) == 0:
            print(f"[SKIP] No face detected: {filename}")
            return
        if len(faces) > 1:
            print(f"[SKIP] Multiple faces ({len(faces)}) in {filename}")
            return

        face = faces[0]

        # Step 5: Align & Embed
        try:
            aligned_face = self.aligner.align(img, face["kps"])
            embedding = self.embedder.get_embedding(aligned_face)
        except Exception as e:
            print(f"[SKIP] Face processing failed for {filename}: {e}")
            return

        # Step 6: Create Person using the correct class_id
        person_id = self.db.get_or_create_person(roll_number, name, class_id=class_id)

        # Step 7: Insert Anchor embedding
        self.db.insert_embedding(
            person_id=person_id,
            embedding=embedding,
            image_path=image_ref,
            type="ANCHOR",
            quality_score=1000
        )
        self.db.update_person_timestamp(person_id)

        print(f"[OK] Enrolled {roll_number} ({name}) into {batch}-{section}")


    def run(self):
        print("=" * 50)
        print("FACE ENROLLMENT SERVICE STARTED")
        print("=" * 50)

        if not os.path.isdir(self.image_dir):
            print(f"[ERROR] Folder not found: {self.image_dir}")
            return

        # --- PASS 1: Process root-level images (STAFF) ---
        root_images = [
            f for f in os.listdir(self.image_dir)
            if os.path.isfile(os.path.join(self.image_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if root_images:
            print(f"[ENROLL] Found {len(root_images)} root-level images. Assigning to STAFF-GENERAL.")
            staff_class_id = self.db.get_or_create_class("STAFF", "GENERAL")
            
            for filename in root_images:
                image_path = os.path.join(self.image_dir, filename)
                self._enroll_single_image(image_path, staff_class_id, "STAFF", "GENERAL")

        # --- PASS 2: Iterate through subfolders (e.g., S4AIE-A) ---
        for folder_name in os.listdir(self.image_dir):
            folder_path = os.path.join(self.image_dir, folder_name)
            
            # Skip if it's a file (we already handled root files above)
            if not os.path.isdir(folder_path):
                continue
                
            # Parse batch and section from folder name
            if "-" not in folder_name:
                print(f"[SKIP] Invalid folder format (expected Batch-Section): {folder_name}")
                continue
                
            batch, section = folder_name.split("-", 1)
            
            # Ensure class exists in DB and get its ID
            class_id = self.db.get_or_create_class(batch, section)
            
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            
            print(f"\n[ENROLL] Found {len(image_files)} images in Class: {batch}-{section}")
            
            for filename in image_files:
                image_path = os.path.join(folder_path, filename)
                self._enroll_single_image(image_path, class_id, batch, section)

        print("=" * 50)
        print("FACE ENROLLMENT COMPLETED")
        print("=" * 50)
        self.db.close()