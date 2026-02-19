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

        # FALSE SAFE BLOCK
        cur.execute("""
                    INSERT INTO classes (id, batch, section)
                    VALUES (1, 'DEFAULT', '0')
                    ON CONFLICT (id) DO NOTHING;
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
        try:
            cur.execute("CREATE TYPE template_type AS ENUM ('ANCHOR', 'ADAPTIVE');")
        except Exception:
            self.conn.rollback()  # In case the type already exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_templates(
                id SERIAL PRIMARY KEY,
                person_id INT REFERENCES persons(id) ON DELETE CASCADE,
                embedding vector(512) NOT NULL,
                type template_type DEFAULT 'ADAPTIVE',
                -- quality & mannagement metadata
                quality_score FLOAT DEFAULT 0.0,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                last_matched_at TIMESTAMP DEFAULT NOW()
                    );
        """)

        # Enabling HNSW for vector indexing
        cur.execute("""
            CREATE INDEX IF NOT EXISTS face_templates_hnsw_idx ON
            face_templates USING hnsw(embedding vector_cosine_ops);
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
                print(f"[SYNC] Linked {batch}-{section} (ID {class_id}) to {subject} on Day {day}")

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
        
    def get_scheduler_state(self):
        """
        Returns a dictionary with 'current' and 'next' session details.
        Used by SessionController to plan transitions.
        """
        now = datetime.now()
        current_day = now.isoweekday()
        current_time = now.time()
        
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Get CURRENT Active Session (if any)
        cur.execute("""
            SELECT cs.class_id, c.batch, c.section, cs.start_time, cs.end_time 
            FROM class_schedule cs
            JOIN classes c ON cs.class_id = c.id
            WHERE cs.day_of_week = %s 
            AND %s BETWEEN cs.start_time AND cs.end_time
            LIMIT 1;
        """, (current_day, current_time))
        current_session = cur.fetchone()
        
        # 2. Get NEXT Upcoming Session
        # We look for the first class that starts AFTER now
        cur.execute("""
            SELECT cs.class_id, c.batch, c.section, cs.start_time, cs.end_time 
            FROM class_schedule cs
            JOIN classes c ON cs.class_id = c.id
            WHERE cs.day_of_week = %s 
            AND cs.start_time > %s
            ORDER BY cs.start_time ASC
            LIMIT 1;
        """, (current_day, current_time))
        next_session = cur.fetchone()
        
        cur.close()
        
        return {
            "current": current_session, # None if we are in a break
            "next": next_session        # None if school is over for the day
        }

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

    # class_id = 1 should be removed later when the class_id will be added to the person creation form in the frontend and the create_person function will be called with the correct class_id from the frontend instead of hardcoding it here
    def create_person(self, roll_number, name, class_id=1):
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
    
    # -- not sure whether this is useful or not let it be later if not used we will remove it --
    def get_or_create_person(self, roll_number, name, class_id=1):
        # NOTE: Updates might be needed later to handle class_id assignment
        person = self.get_person_by_roll(roll_number)
        if person:
            return person[0]  # id
        return self.create_person(roll_number, name, class_id)
    
    # this funciton is for updating the ID card image if a student want to change the or update the ID card image at that case we should run this fun
    def update_person_timestamp(self, person_id):
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE persons SET updated_at=NOW() WHERE id=%s;",
            (person_id,)
        )
        self.conn.commit()

    def insert_embedding(self, person_id, embedding, image_ref, type ='ADAPTIVE', quality_score=0.0):
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO face_templates
                (person_id, embedding, image_ref, type, quality_score, created_at, last_matched_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id;
                """,
                (person_id, embedding.tolist(), image_ref, type, quality_score)
            )
            new_id = cur.fetchone()[0]
            self.conn.commit()
            return new_id
        except Exception as e:
            self.conn.rollback()
            print(f"[DB ERROR] Insert Template Failed: {e}")
            return None
        
    def get_gallery_by_class(self, class_id):
        """
        Docstring for get_gallery_by_class
        retrive the full gallery of a class by taking class_id as input
        :param self: Description
        :param class_id: Description 
        """
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT
                ft.person_id,
                p.name,
                p.roll_number,
                ft.embedding,
                ft.type,
                ft.id as template_id
            FROM face_templates ft
            JOIN persons p ON ft.person_id = p.id;
        """
        if class_id:
            query += " WHERE p.class_id = %s;"
            params = (class_id,)
        else:
            params = ()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        gallery = {}
        for row in rows:
            pid = row['person_id']
            if pid not in gallery:
                gallery[pid] = {
                    "name": row['name'],
                    "roll_number": row['roll_number'],
                    "templates": []
                }
            gallery[pid]['templates'].append({
                'id': row['template_id'],
                'embedding': row['embedding'],
                'type': row['type']
            })
        return gallery

    # This function will be used to update the template usage metadata after each attendance marking so that we can implement smarter template management strategies in the future (like retiring old templates, promoting good templates to anchors, etc.)
    def update_template_usage(self, template_id):
        cur = self.conn.cursor()
        cur.execute("UPDATE face_templates SET last_matched_at = NOW() WHERE id = %s;", (template_id,))
        self.conn.commit()
    
    def delete_poor_template(self, person_id):
        # Deletes the worst adaptive template lowest quality or oldest ones
        curr = self.conn.cursor()
        try:
            curr.execute("""
                DELETE FROM face_templates
                WHERE id IN (
                    SELECT id FROM face_templates
                    WHERE person_id = %s AND type = 'ADAPTIVE'
                    ORDER BY quality_score ASC, last_matched_at ASC
                    LIMIT 1
                )
                RETURNING image_path; 
            """, (person_id,))
            deleted = curr.fetchone()
            self.conn.commit()
            return deleted[0] if deleted else None
        except Exception as e:
            self.conn.rollback()
            print(f"[DB ERROR] Delete Template Failed: {e}")
            return None
           
    def get_person_count(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM persons;")
        count = cur.fetchone()[0]
        cur.close()
        return count
    
    # whats the use, if not used should be removed later
    def image_ref_exists(self, image_ref):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM face_templates WHERE image_path=%s LIMIT 1;",
            (image_ref,)
        )
        return cur.fetchone() is not None
    
    def get_embeddings_by_person(self, person_id):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT embedding FROM face_templates WHERE person_id=%s;",
            (person_id,)
        )
        rows = cur.fetchall()
        return [row[0] for row in rows]

    # --------------------------------------------------------------
    # ATTENDANCE FUNCTIONS
    # --------------------------------------------------------------

    def insert_attendance(self, person_id, track_id, confidence, face_crop_path = None):
        cur = self.conn.cursor()
        current_ts = datetime.now()
        query = """
            INSERT INTO attendance_log (person_id, confidence, track_id, face_crop_path, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (person_id, date)
            DO UPDATE SET
                confidence = EXCLUDED.confidence,
                face_crop_path = EXCLUDED.face_crop_path,
                timestamp = EXCLUDED.timestamp
            WHERE EXCLUDED.confidence > attendance_log.confidence;
        """
        try:
            cur.execute(query, (person_id, confidence, track_id, face_crop_path, current_ts))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[DB ERROR] Insert Attendance Failed: {e}")
        finally:
            cur.close()

        self.conn.commit()

    # --------------------------------------------------------------

    def close(self):
        if self.conn:
            self.conn.close()
            print("[DB] Connection closed.")
