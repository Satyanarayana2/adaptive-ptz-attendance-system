# core/recognizer.py

import numpy as np


class Recognizer:
    """
    Face Recognizer for the attendance system.

    Responsibilities:
      - Take ALIGNED face
      - Generate embedding
      - Query PostgreSQL for best match using pgvector
      - Return best matched PERSON (not just embedding)
    """

    def __init__(self, embedder, db, threshold=0.50):
        self.embedder = embedder
        self.db = db
        self.threshold = threshold

    # ----------------------------------------------------

    def recognize(self, aligned_face):
        """
        Input:
            aligned_face : aligned face image

        Output:
            dict with recognition result
        """

        # 1. Generate embedding
        live_embedding = self.embedder.get_embedding(aligned_face)
        if live_embedding is None:
            return {
                "matched": False,
                "person_id": None,
                "name": None,
                "score": 0.0,
                "embedding": None
            }

        # 2. Query DB for best match
        result = self._find_best_match(live_embedding)
        if result is None:
            return {
                "matched": False,
                "person_id": None,
                "name": None,
                "score": 0.0,
                "embedding": live_embedding
            }

        person_id, name, roll_number, score = result

        # 3. Decide match
        if score >= self.threshold:
            return {
                "matched": True,
                "person_id": person_id,
                "name": name,
                "score": float(score),
                "embedding": live_embedding
            }

        return {
            "matched": False,
            "person_id": None,
            "name": None,
            "score": float(score),
            "embedding": live_embedding
        }

    # ----------------------------------------------------

    def _find_best_match(self, query_embedding):
            """
            Query PostgreSQL for the single closest matching embedding.
            Uses HNSW index for O(log N) speed.
            """
            cur = self.db.conn.cursor()
            
            # Fast Nearest Neighbor Query
            query = """
                SELECT p.id, p.name, p.roll_number, 
                    (1 - (fe.embedding <=> %s)) AS similarity
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
                ORDER BY fe.embedding <=> %s ASC
                LIMIT 1;
            """
            
            # Pass the raw query_embedding directly (no strings needed!)
            cur.execute(query, (query_embedding, query_embedding))
            row = cur.fetchone()
            cur.close()
            return row
