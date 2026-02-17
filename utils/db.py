import psycopg2
import json
import os
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from datetime import datetime

class Database:
    """
    PostgreSQL database handler for:
      - persons table (Students)
      - face_embeddings table (The Gallery)
      - attendance_log table
      - classes & class_schedule tables (Zone 1)
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

        # 1. ZONE 1: CLASSES & SCHEDULE
        # The Class Definitions (Who)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id SERIAL PRIMARY KEY,
                batch VARCHAR(10) NOT NULL,
                section VARCHAR(5) NOT NULL,
                CONSTRAINT unique_class_def UNIQUE (batch, section)
            );
        """)

        # The Schedule (When)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS class_schedule (
                id SERIAL PRIMARY KEY,
                class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
                day_of_week INTEGER NOT NULL,
                start_time TIME NOT NULL,
                end_time TIME NOT NULL,
                subject_code VARCHAR(20) NOT NULL
            );
        """)

        # 2. ZONE 2: STUDENTS (PERSONS)
        # Added class_id foreign key to link students to their batch
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                roll_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                class_id INTEGER REFERENCES classes(id) DEFAULT 1, -- need to remove the default after all tests are done
                created_at TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP),
                updated_at TIME(0) DEFAULT date_trunc('minute', CURRENT_TIMESTAMP)
            );
        """)

        # 3. ZONE 3: FACE EMBEDDINGS (The Gallery)
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

        # 4. ATTENDANCE LOG
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance_log (
                id SERIAL PRIMARY KEY,
                person_id INT REFERENCES persons(id),
                date DATE DEFAULT CURRENT_DATE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Fixed: Using TIMESTAMP for precision
                confidence FLOAT,
                face_crop_path TEXT,
                track_id INT,
                source TEXT DEFAULT 'webcam',
                UNIQUE(person_id, date)
            );
        """)

        self.conn.commit()
        print("[DB] Tables ensured.")

    # --------------------------------------------------------------
    # ZONE 1 FUNCTIONS: TIMETABLE MANAGEMENT
    # --------------------------------------------------------------

    def sync_timetable(self, json_file_path):
        """
        Safely syncs the database.
        1. Validates JSON structure first (Strict Mode).
        2. If valid: Replaces the old schedule.
        3. If invalid: ROLLS BACK and keeps the old schedule.
        """
        try:
            with open(json_file_path, 'r') as f:
                timetable_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read JSON file: {e}")
            return

        cursor = self.conn.cursor()
        required_keys = {'day_of_week', 'start_time', 'end_time', 'course_code', 'batch', 'section'}

        try:
            # --- STEP 1: PRE-VALIDATION ---
            # Check every entry BEFORE we even start the transaction.
            for i, entry in enumerate(timetable_data):
                if not required_keys.issubset(entry.keys()):
                    missing = required_keys - entry.keys()
                    # The exact error message you requested:
                    raise ValueError(f"Can't map the timetable you provided. (Error in Entry #{i+1}: Missing keys {missing})")

            # --- STEP 2: TRANSACTION START ---
            print("[SYNC] Validation passed. Updating schedule...")
            
            # Clear old schedule (Will be undone if code crashes later)
            cursor.execute("TRUNCATE TABLE class_schedule RESTART IDENTITY CASCADE;")

            # --- STEP 3: INGESTION ---
            for entry in timetable_data:
                batch = entry['batch']
                subject = entry['course_code']
                section = entry['section']
                day = entry['day_of_week']
                start = entry['start_time']
                end = entry['end_time']

                # Get or create class_id
                cursor.execute("""
                    INSERT INTO classes (batch, section)
                    VALUES (%s, %s)
                    ON CONFLICT (batch, section)
                    DO UPDATE SET batch=EXCLUDED.batch
                    RETURNING id;
                """, (batch, section))

                class_id_row = cursor.fetchone()
                if not class_id_row:
                    raise ValueError(f"Failed to get or create class for batch {batch} and section {section}.")
                class_id = class_id_row[0]
                # Insert schedule entry
                cursor.execute("""
                    INSERT INTO class_schedule (class_id, day_of_week, start_time, end_time, subject_code)
                    VALUES (%s, %s, %s, %s, %s)""", (class_id, day, start, end, subject))
                print(f"[SYNC] Added: {batch} {section} - {subject} on Day {day} from {start} to {end}")

            # --- STEP 4: COMMIT ---
            self.conn.commit()
            print("[SUCCESS] Timetable synced successfully.")

        except ValueError as ve:
            # Verification Failed (Expected Error)
            self.conn.rollback()
            print(f"[ERROR] {ve}") 
            print("[SAFEGUARD] Database was NOT modified. Old schedule is active.")

        except Exception as e:
            # Database/System Error (Unexpected)
            self.conn.rollback()
            print(f"[CRITICAL ERROR] Timetable Sync Failed! {e}")
            print("[SAFEGUARD] Rolling back to previous schedule.")
            
        finally:
            cursor.close()

    def get_current_class(self):
        """
        Returns the class_id active RIGHT NOW based on system time.
        """
        now = datetime.now()
        current_day = now.isoweekday() # 1=Mon, 7=Sun
        current_time = now.time()

        cur = self.conn.cursor()
        cur.execute("""
            SELECT class_id FROM class_schedule 
            WHERE day_of_week = %s 
            AND %s BETWEEN start_time AND end_time
            LIMIT 1;
        """, (current_day, current_time))
        
        result = cur.fetchone()
        cur.close()
        return result[0] if result else None

    # --------------------------------------------------------------
    # ZONE 2 & 3 FUNCTIONS: PERSONS & EMBEDDINGS
    # --------------------------------------------------------------

    def get_person_by_roll(self, roll_number):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, roll_number, name FROM persons WHERE roll_number=%s;",
            (roll_number,)
        )
        return cur.fetchone()

    def create_person(self, roll_number, name, class_id=None):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO persons (roll_number, name, class_id, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
            RETURNING id;
            """,
            (roll_number, name, class_id)
        )
        person_id = cur.fetchone()[0]
        self.conn.commit()
        return person_id
    
    def get_or_create_person(self, roll_number, name):
        # NOTE: Updates might be needed later to handle class_id assignment
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
    
    def get_all_embeddings(self, class_id=None):
        """
        Retrieves embeddings. 
        If class_id is provided (Time-Aware Mode), only returns students from that class.
        """
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        if class_id:
            query = """
                SELECT 
                    p.id AS person_id,
                    p.name,
                    p.roll_number,
                    fe.embedding
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
                WHERE p.class_id = %s;
            """
            cur.execute(query, (class_id,))
        else:
            query = """
                SELECT 
                    p.id AS person_id,
                    p.name,
                    p.roll_number,
                    fe.embedding
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id;
            """
            cur.execute(query)
            
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

    # --------------------------------------------------------------
    # ATTENDANCE FUNCTIONS
    # --------------------------------------------------------------

    def insert_attendance(self, person_id, confidence, track_id, source="webcam", face_crop_path=None):
        """Record attendance with metadata. Corrected timestamp logic."""
        cur = self.conn.cursor()
        
        # Use Python timestamp to ensure DB and File sync match perfectly
        current_ts = datetime.now()

        query = """
            INSERT INTO attendance_log (person_id, confidence, track_id, source, face_crop_path, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id, date) 
            DO UPDATE SET 
                confidence = EXCLUDED.confidence,
                face_crop_path = EXCLUDED.face_crop_path,
                timestamp = EXCLUDED.timestamp
            WHERE EXCLUDED.confidence > attendance_log.confidence;
        """
        try:
            cur.execute(query, (person_id, confidence, track_id, source, face_crop_path, current_ts))
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
