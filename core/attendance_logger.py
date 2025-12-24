# core/attendance_logger.py

import time


class AttendanceLogger:
    """
    Handles high-level attendance logic:
      - prevents duplicate logs for same person
      - uses cooldown period
      - interacts with DB for actual insertion
    """

    def __init__(self, db, cooldown_seconds=20):
        self.db = db
        self.cooldown = cooldown_seconds

        # person_id → last_mark_timestamp
        self.last_mark_time = {}

    # -------------------------------------------------------------

    def mark_attendance(self, person_id, confidence, track_id, source="webcam"):
        """
        Marks attendance if cooldown period has passed.
        Returns:
            True  → attendance logged
            False → skipped (duplicate within cooldown window)
        """

        now = time.time()

        # Check last marked time
        if person_id in self.last_mark_time:
            elapsed = now - self.last_mark_time[person_id]

            if elapsed < self.cooldown:
                # Skip duplicate log
                print(f"[ATTENDANCE] Skipped duplicate for person_id={person_id}")
                return False

        # Log into DB
        self.db.insert_attendance(
            person_id=person_id,
            confidence=confidence,
            track_id=track_id,
            source=source
        )

        # Update local memory
        self.last_mark_time[person_id] = now

        print(f"[ATTENDANCE] Marked for person_id={person_id}")
        return True

    # -------------------------------------------------------------
