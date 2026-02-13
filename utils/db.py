# utils/db.py


import psycopg2
import json
import os
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

class Database:
    """
    PostgreSQL database handler for:
      - persons table
      - face_embeddings table
      - attendance_log table
    """

    def __init__(self, config_path="config/db_config.json"):
        self.config_path = config_path
        self.conn = None
        self._connect()
        self._enable_vector_extension()
        register_vector(self.conn)
        self._create_tables()

    # --------------------------------------------------------------

    def _enable_vector_extension(self):
        curr = self.conn.cursor()
        try:
            curr.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            print("[DB] Vector extension enabled.")
        except Exception as e:
            self.conn.rollback()
            print(f"[DB] Error enabling vector extension: {e}")
        finally:
            curr.close()

    def _connect(self):
            """Connect to PostgreSQL with simple retries."""
            import time
            max_retries = 5
            for i in range(max_retries):
                try:
                    with open(self.config_path, "r") as f:
                        cfg = json.load(f)
                    self.conn = psycopg2.connect(
                        host=cfg["host"],
                        port=cfg["port"],
                        user=cfg["user"],
                        password=cfg["password"],
                        database=cfg["database"]
                    )
                    print("[DB] Connected to PostgreSQL.")
                    return
                except Exception as e:
                    print(f"[DB] Connection failed (Attempt {i+1}/{max_retries}): {e}")
                    time.sleep(3)
            raise Exception("Could not connect to database after retries.")

    # --------------------------------------------------------------

    def _create_tables(self):
        """Create tables if not existing."""
        cur = self.conn.cursor()

        # persons table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                roll_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP),
                updated_at TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP)
            );
        """)

        # face embeddings table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id SERIAL PRIMARY KEY,
                person_id INT REFERENCES persons(id) ON DELETE CASCADE,
                embedding VECTOR(512) NOT NULL,
                image_ref TEXT,
                created_at TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP)
            );
        """)

        # Enabling HNSW for vector indexing
        cur.execute("""
                    CREATE INDEX IF NOT EXISTS face_embeddings_hnsw_idx ON
                    face_embeddings USING hnsw(embedding vector_cosine_ops);
                    """)

        # attendance table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance_log (
                id SERIAL PRIMARY KEY,
                person_id INT REFERENCES persons(id),
                date DATE DEFAULT CURRENT_DATE,
                timestamp TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP),
                confidence FLOAT,
                face_crop_path TEXT,
                track_id INT,
                source TEXT DEFAULT 'webcam',
                UNIQUE(person_id, date)
            );
        """)

        # Time Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS timetable_slots (
                    id SERIAL PRIMARY KEY,
                    day_of_week INT NOT NULL,
                    start_time TIME NOT NULL,
                    end_time TIME NOT NULL,
                    course_code TEXT NOT NULL,
                    batch TEXT,
                    section TEXT, 
                    lab_name TEXT DEFAULT 'Hardware & IoT Lab',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(day_of_week, start_time, end_time)
                    );
        """)

        self.conn.commit()
        print("[DB] Tables ensured.")

    # PERSON FUNCTIONS

    def get_person_by_roll(self, roll_number):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, roll_number, name FROM persons WHERE roll_number=%s;",
            (roll_number,)
        )
        return cur.fetchone()

    def create_person(self, roll_number, name):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO persons (roll_number, name, created_at, updated_at)
            VALUES (%s, %s, NOW(), NOW())
            RETURNING id;
            """,
            (roll_number, name)
        )
        person_id = cur.fetchone()[0]
        self.conn.commit()
        return person_id
    
    def get_or_create_person(self, roll_number, name):
        person = self.get_person_by_roll(roll_number)

        if person:
            return person[0]  # id

        return self.create_person(roll_number, name)
    
    def update_person_timestamp(self, person_id):
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE persons SET updated_at=NOW() WHERE id=%s;",
            (person_id,)
        )
        self.conn.commit()

    # EMBEDDING FUNCTIONS

    def insert_embedding(self, person_id, embedding, image_ref):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO face_embeddings (person_id, embedding, image_ref, created_at)
            VALUES (%s, %s, %s, NOW());
            """,
            (person_id, embedding.tolist(), image_ref)
        )
        self.conn.commit()
    
    def get_all_embeddings(self):
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT 
                p.id AS person_id,
                p.name,
                p.roll_number,
                fe.embedding
            FROM face_embeddings fe
            JOIN persons p ON fe.person_id = p.id;
        """)
        rows = cur.fetchall()
        cur.close()
        return rows
    
    def get_person_count(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM persons;")
        count = cur.fetchone()[0]
        cur.close()
        return count
    
    def image_ref_exists(self, image_ref):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM face_embeddings WHERE image_ref=%s LIMIT 1;",
            (image_ref,)
        )
        return cur.fetchone() is not None
    
    def get_embeddings_by_person(self, person_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT embedding FROM face_embeddings WHERE person_id=%s;",
            (person_id,)
        )
        rows = cur.fetchall()
        return [row[0] for row in rows]


    # ATTENDANCE FUNCTIONS

    def insert_attendance(self, person_id, confidence, track_id, source="webcam", face_crop_path=None):
        """Record attendance with metadata."""
        cur = self.conn.cursor()

        query = """
                INSERT INTO attendance_log (person_id, confidence, track_id, source, face_crop_path)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (person_id, date) 
                DO UPDATE SET 
                    confidence = EXCLUDED.confidence,
                    face_crop_path = EXCLUDED.face_crop_path,
                    timestamp = EXCLUDED.timestamp
                WHERE EXCLUDED.confidence > attendance_log.confidence;
                """
        try:
            cur.execute(query, (person_id, confidence, track_id, source, face_crop_path))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[DB ERROR] Attendance insertion failed: {e}")
        finally:
            cur.close()

        self.conn.commit()

    # --------------------------------------------------------------

    def close(self):
        if self.conn:
            self.conn.close()
            print("[DB] Connection closed.")
