# core/recognizer.py

import numpy as np
from psycopg2.extras import RealDictCursor

class Recognizer:
    """
    Stateless Face Recognizer.
    
    Logic:
      1. Generates the live embedding.
      2. Sends it directly to the PostgreSQL database.
      3. The DB dynamically filters by the active class schedule.
      4. The DB performs Nearest Neighbor comparison across Anchors and Adaptives.
      5. Returns the single Best Match.
    """

    def __init__(self, embedder, db, threshold=0.50):
        self.embedder = embedder
        self.db = db
        # Python uses Similarity (Higher is better), pgvector uses Distance (Lower is better)
        self.similarity_threshold = threshold
        self.distance_threshold = 1.0 - threshold

    def recognize(self, aligned_face):
        """
        Compares the aligned face directly against the database using pgvector.
        """
        # 1. Generate Embedding
        live_vector = self.embedder.get_embedding(aligned_face)
        if live_vector is None:
            return self._empty_result()

        # 2. Database Search Query
        # This query perfectly mimics the old Python logic: 
        # - It dynamically finds the active class.
        # - If active, it filters by that class. If not active, it searches everyone.
        # - It calculates Cosine Distance (<=>) across ALL templates (Anchor + Adaptive).
        # - It returns the single closest match (ORDER BY distance ASC LIMIT 1).
        
        query = """
        WITH current_schedule AS (
            -- Find the currently active class_id based on day and time
            SELECT class_id 
            FROM class_schedule
            WHERE day_of_week = EXTRACT(ISODOW FROM CURRENT_TIMESTAMP)
            AND CURRENT_TIME BETWEEN start_time AND end_time
            LIMIT 1
        )
        SELECT 
            p.id AS person_id,
            p.name,
            p.roll_number,
            ft.id AS template_id,
            (ft.embedding <=> %s::vector) AS distance
        FROM face_templates ft
        JOIN persons p ON ft.person_id = p.id
        WHERE 
            -- FILTER LOGIC:
            -- If current_schedule has a class, only match persons in that class.
            -- If current_schedule is empty (No class running), match ALL persons.
            (
                (SELECT class_id FROM current_schedule) IS NULL 
                OR 
                p.class_id = (SELECT class_id FROM current_schedule)
            )
            -- THRESHOLD LOGIC: Only consider matches closer than our threshold
            AND (ft.embedding <=> %s::vector) <= %s
        ORDER BY distance ASC
        LIMIT 1;
        """

        try:
            cur = self.db.conn.cursor(cursor_factory=RealDictCursor)
            
            # We must pass the live_vector twice (once for the SELECT calculation, once for the WHERE filter)
            vector_str = str(live_vector.tolist())
            cur.execute(query, (vector_str, vector_str, self.distance_threshold))
            
            match = cur.fetchone()
            cur.close()

            # 3. Process Result
            if match:
                # Convert DB Distance back to Python Similarity
                similarity_score = 1.0 - match['distance']
                
                return {
                    "matched": True,
                    "person_id": match['person_id'],
                    "name": match['name'],
                    "roll": match['roll_number'],
                    "score": float(similarity_score),
                    "matched_template_id": match['template_id'],
                    "embedding": live_vector
                }
                
        except Exception as e:
            print(f"[RECOGNIZER DB ERROR] {e}")
            self.db.conn.rollback()

        # If no match found or error occurred
        return self._empty_result(embedding=live_vector)

    def _empty_result(self, score=0.0, embedding=None):
        return {
            "matched": False,
            "person_id": None,
            "name": None,
            "roll": None,
            "score": float(score),
            "matched_template_id": None,
            "embedding": embedding
        }