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
            ft.type AS template_type,
            (1 - (ft.embedding <=> %s::vector)) AS similarity
        FROM face_templates ft
        JOIN persons p ON ft.person_id = p.id
        JOIN classes c ON p.class_id = c.id
        WHERE 
            (
                (SELECT class_id FROM current_schedule) IS NULL 
                OR 
                p.class_id = (SELECT class_id FROM current_schedule)
                OR
                c.batch = 'STAFF'
            )
        -- No threshold filter here — applied in Python so we get best_score for unknowns
        ORDER BY (ft.embedding <=> %s::vector) ASC
        LIMIT 1;
        """

        try:
            cur = self.db.conn.cursor(cursor_factory=RealDictCursor)
            vector_str = str(live_vector.tolist())
            
            # 2 params: SELECT similarity, ORDER BY distance
            cur.execute(query, (
                vector_str,
                vector_str
            ))
            
            match = cur.fetchone()
            cur.close()

            if match:
                sim_score = float(match['similarity'])

                if sim_score >= self.similarity_threshold:
                    print(f"[DEBUG] DB matched {match['name']} with score: {sim_score:.4f}")
                    return {
                        "matched": True,
                        "person_id": match['person_id'],
                        "name": match['name'],
                        "roll": match['roll_number'],
                        "score": sim_score,
                        "matched_template_id": match['template_id'],
                        "matched_template_type": match['template_type'],
                        "embedding": live_vector
                    }
                else:
                    # Below threshold — known person at bad angle, or too far
                    # Return best_score so caller can decide whether to save Unknown crop
                    print(f"[DEBUG] Below threshold: best match {match['name']} at {sim_score:.4f} (need {self.similarity_threshold})")
                    return self._empty_result(score=sim_score, embedding=live_vector)
            else:
                # No face templates in DB at all for this class
                print(f"[DEBUG] No faces in DB for current schedule")
                
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