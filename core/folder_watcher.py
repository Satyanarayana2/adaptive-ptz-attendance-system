import os
import cv2

from utils.db import Database
from utils.detectors.insight_detector import InsightDetector
from utils.embeddings.insight_embedder import InsightEmbedder
from utils.face_alignment import FaceAligner


class FolderWatcher:
    """
    Offline face enrollment service.

    Reads images from a folder, detects exactly one face,
    aligns it, generates embeddings, and stores them in DB.
    class_id is set as 1 in db side query for just time being later 
    should add related changes to fetech the class_id according to the 
    folder or student images file names. REMEMBER TO DO THIS AFTER TESTS.
    """

    def __init__(self, image_dir="Face_images"):
        self.image_dir = image_dir

        # DB
        self.db = Database()

        # Detector
        self.detector = InsightDetector()
        self.detector.prepare()

        # Aligner + Embedder
        self.aligner = FaceAligner()
        self.embedder = InsightEmbedder()
        self.embedder.prepare()

    def _parse_filename(self, filename):
        """
        Expected format:
        AM.SC.P2ARI24009_SATYA.jpg

        Returns:
            roll_number, name
        """
        name_part = os.path.splitext(filename)[0]

        if "_" not in name_part:
            return None, None

        parts = name_part.split("_", 1)
        return parts[0], parts[1]

    def run(self):
        print("=" * 50)
        print("FACE ENROLLMENT SERVICE STARTED")
        print("=" * 50)

        if not os.path.isdir(self.image_dir):
            print(f"[ERROR] Folder not found: {self.image_dir}")
            return

        image_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"[ENROLL] Found {len(image_files)} images")

        for filename in image_files:
            image_path = os.path.join(self.image_dir, filename)
            image_ref = os.path.relpath(image_path)

            # STEP 1 — check duplicate image
            if self.db.image_ref_exists(image_ref):
                print(f"[SKIP] Already enrolled: {filename}")
                continue

            # STEP 2 — parse filename
            roll_number, name = self._parse_filename(filename)
            if not roll_number or not name:
                print(f"[SKIP] Invalid filename format: {filename}")
                continue

            # STEP 3 — load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"[SKIP] Cannot read image: {filename}")
                continue

            # STEP 4 — detect faces
            faces = self.detector.detect(img)

            if len(faces) == 0:
                print(f"[SKIP] No face detected: {filename}")
                continue

            if len(faces) > 1:
                print(f"[SKIP] Multiple faces ({len(faces)}) in {filename}")
                continue

            face = faces[0]

            # STEP 5 — align face
            try:
                aligned_face = self.aligner.align(img, face["kps"])
            except Exception as e:
                print(f"[SKIP] Alignment failed for {filename}: {e}")
                continue

            # STEP 6 — generate embedding
            try:
                embedding = self.embedder.get_embedding(aligned_face)
            except Exception as e:
                print(f"[SKIP] Embedding failed for {filename}: {e}")
                continue

            # STEP 7 — ensure person
            person_id = self.db.get_or_create_person(roll_number, name)

            # STEP 8 — insert embedding
            self.db.insert_embedding(
                person_id=person_id,
                embedding=embedding,
                image_path=image_ref,
                type="ANCHOR",
                quality_score=1000
            )

            # STEP 9 — update person timestamp
            self.db.update_person_timestamp(person_id)

            print(f"[OK] Enrolled {roll_number} ({name}) from {filename}")

        print("=" * 50)
        print("FACE ENROLLMENT COMPLETED")
        print("=" * 50)
        self.db.close()