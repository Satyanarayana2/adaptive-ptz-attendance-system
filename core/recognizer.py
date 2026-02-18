# core/recognizer.py

import numpy as np


class Recognizer:
    """
    Timetable-Aware Face Recognizer (In-Memory).
    
    Logic:
      1. Loads ONLY the current class's faces into RAM (reload_gallery).
      2. Compares live face against ALL templates (Anchor + Adaptives) for those students.
      3. Returns the MAX score (Best Match).
    """

    def __init__(self, embedder, db, threshold=0.50):
        self.embedder = embedder
        self.db = db
        self.threshold = threshold
        
        # The In-Memory Gallery
        # Structure: { person_id: {'name': str, 'roll': str, 'templates': [vectors...]} }
        self.known_faces = {} 
        self.loaded_class_id = -999 # Tracks which class is currently loaded

    def reload_gallery(self, class_id):
        """
        Called by main.py when the class changes.
        Loads the specific student list for this schedule.
        """
        print(f"[RECOGNIZER] Loading Gallery for Class ID: {class_id}...")
        
        # Fetch from DB (using the updated function in db.py)
        # Returns dict: { person_id: { 'name':..., 'templates': [{'embedding':...}, ...] } }
        self.known_faces = self.db.get_gallery_by_class(class_id)
        self.loaded_class_id = class_id
        
        # Stats for logs
        student_count = len(self.known_faces)
        total_templates = sum(len(data['templates']) for data in self.known_faces.values())
        print(f"[RECOGNIZER] Memory Updated: {student_count} students, {total_templates} templates loaded.")

    def recognize(self, aligned_face):
        """
        Compares the aligned face against the loaded in-memory gallery.
        Uses Max-Pooling (Nearest Neighbor) strategy.
        """
        # 1. Generate Embedding (Using InsightFace)
        live_vector = self.embedder.get_embedding(aligned_face)
        if live_vector is None:
            return self._empty_result()

        # 2. In-Memory Search (Max-Pooling)
        best_match = {
            "person_id": None,
            "score": 0.0,
            "template_id": None,
            "name": None,
            "roll": None
        }

        # Loop through every student currently in memory
        for person_id, data in self.known_faces.items():
            
            # Check against ALL their templates (Anchor + Adaptives)
            for tmpl in data['templates']:
                # Convert to numpy for math
                stored_vector = np.array(tmpl['embedding'], dtype=np.float32)
                
                # Calculate Cosine Similarity (Dot Product since vectors are L2 normalized)
                score = np.dot(live_vector, stored_vector)
                
                # Keep the highest score found so far
                if score > best_match["score"]:
                    best_match["score"] = score
                    best_match["person_id"] = person_id
                    best_match["template_id"] = tmpl['id']
                    best_match["name"] = data['name']
                    best_match["roll"] = data['roll']

        # 3. Threshold Decision
        if best_match["score"] >= self.threshold:
            return {
                "matched": True,
                "person_id": best_match["person_id"],
                "name": best_match["name"],
                "score": float(best_match["score"]),
                "matched_template_id": best_match["template_id"], # Critical for Zone 3 Logic
                "embedding": live_vector # Return this so we can save it if it's a new adaptive face
            }
        
        return self._empty_result(score=best_match["score"], embedding=live_vector)

    def _empty_result(self, score=0.0, embedding=None):
        return {
            "matched": False,
            "person_id": None,
            "name": None,
            "score": float(score),
            "matched_template_id": None,
            "embedding": embedding
        }