# core/recognizer.py

import numpy as np
from psycopg2.extras import RealDictCursor

class Recognizer:
    """
    Stateless Face Recognizer.
    """

    def __init__(self, embedder, db, threshold=0.50):
        self.embedder = embedder
        self.db = db
        # We only need the similarity threshold now!
        self.similarity_threshold = float(threshold)

    def recognize(self, aligned_face):
        # 1. Generate Embedding
        live_vector = self.embedder.get_embedding(aligned_face)
        if live_vector is None:
            return self._empty_result()

        # 2. Database Search Query
        query = """
        WITH current_schedule AS (
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
            (1 - (ft.embedding <=> %s::vector)) AS similarity
        FROM face_templates ft
        JOIN persons p ON ft.person_id = p.id
        WHERE 
            (
                (SELECT class_id FROM current_schedule) IS NULL 
                OR 
                p.class_id = (SELECT class_id FROM current_schedule)
            )
            -- Apply SIMILARITY threshold directly (e.g., 0.35)
            AND (1 - (ft.embedding <=> %s::vector)) >= %s
            
        -- Order by raw distance ASC (smallest distance = highest similarity)
        ORDER BY (ft.embedding <=> %s::vector) ASC
        LIMIT 1;
        """

        try:
            cur = self.db.conn.cursor(cursor_factory=RealDictCursor)
            vector_str = str(live_vector.tolist())
            
            # Pass parameters: SELECT vector, WHERE vector, Threshold, ORDER BY vector
            cur.execute(query, (
                vector_str, 
                vector_str, 
                self.similarity_threshold, 
                vector_str
            ))
            
            match = cur.fetchone()
            cur.close()

            if match:
                sim_score = float(match['similarity'])
                print(f"[DEBUG] DB matched {match['name']} with score: {sim_score:.4f}")
                
                return {
                    "matched": True,
                    "person_id": match['person_id'],
                    "name": match['name'],
                    "roll": match['roll_number'],
                    "score": sim_score,
                    "matched_template_id": match['template_id'],
                    "embedding": live_vector
                }
            else:
                print(f"[DEBUG] No match found above threshold {self.similarity_threshold}")
                
        except Exception as e:
            print(f"[RECOGNIZER DB ERROR] {e}")
            self.db.conn.rollback()

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