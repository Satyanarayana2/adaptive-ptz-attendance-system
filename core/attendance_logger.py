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
        
        # track_id → set of person_ids already recognized in this track
        self.recognized_tracks = {}
        
        # {track_id: (person_id, original_score)} cache to skip embedding for already recognized tracks
        # track_id → person_id (recognition cache to skip expensive embedding)
        self.track_recognition_cache = {}
        
        # Stats for monitoring
        self.cache_hits = 0
        self.cache_misses = 0

    # -------------------------------------------------------------

    def get_recognized_person(self, track_id):
        """
        Check if person was already recognized in this track.
        Returns person_id if cached, None if not cached.
        This saves CPU by skipping embedding generation.
        """
        if track_id in self.track_recognition_cache:
            data = self.track_recognition_cache[track_id] # Get cached score
            self.cache_hits += 1
            print(f"[CACHE HIT] Track {track_id} - - > Person {data[0]} (Hits: {self.cache_hits})")
            return data
        else:
            self.cache_misses += 1 
            return None

    def cache_recognition(self, track_id, person_id, original_score=None):
        """
        Cache the recognized person for this track.
        Next frame in same track will use cached result, skipping embedding.
        """
        self.track_recognition_cache[track_id] = (person_id, original_score)
        print(f"[CACHE SET] Cached Person {person_id} for Track {track_id} with score {original_score:.2f}")

    def should_log(self, track_id, person_id):
        """
        Check if we should log attendance for this person in this track.
        Returns True if we should log, False if it's a duplicate within cooldown.
        """
        now = time.time()

        # Check if this person was already recognized in this track
        if track_id in self.recognized_tracks and person_id in self.recognized_tracks[track_id]:
            return False

        # Check cooldown for this person
        if person_id in self.last_mark_time:
            elapsed = now - self.last_mark_time[person_id]
            if elapsed < self.cooldown:
                print(f"[COOLDOWN] Skipped duplicate for person_id={person_id} (cooldown)")
                return False

        return True
    # -------------------------------------------------------------

    def mark_attendance(self, person_id, confidence, track_id, source="webcam", face_crop_path=None):
        """
        Marks attendance using hybrid approach:
        1. Checks if track_id already recognized this person
        2. Checks if person is within cooldown period
        
        Returns:
            True  → attendance logged
            False → skipped (duplicate within cooldown window or same track)
        """

        now = time.time()

        # Check 1: Is this person already recognized in this track?
        if track_id in self.recognized_tracks:
            if person_id in self.recognized_tracks[track_id]:
                print(f"[ATTENDANCE] Person {person_id} already recognized in track {track_id}")
                return False

        # Check 2: Person cooldown (safety net for camera issues)
        if not self.should_log(track_id, person_id):
            return False

        # Log into DB
        self.db.insert_attendance(
            person_id=person_id,
            confidence=confidence,
            track_id=track_id,
            source=source,
            face_crop_path=face_crop_path
        )

        # Update local memory
        self.last_mark_time[person_id] = now
        
        # Track this person in this track
        if track_id not in self.recognized_tracks:
            self.recognized_tracks[track_id] = set()
        self.recognized_tracks[track_id].add(person_id)

        print(f"[ATTENDANCE] Marked for person_id={person_id} in track {track_id}")
        return True

    # -------------------------------------------------------------

    def cleanup_old_tracks(self, current_track_ids):
        """
        Removes tracks from recognition cache that no longer exist.
        Call this periodically to clean up memory.
        
        Args:
            current_track_ids: Set or list of track_ids currently active
        """
        current_set = set(current_track_ids)
        old_tracks = set(self.recognized_tracks.keys()) - current_set
        
        for track_id in old_tracks:
            del self.recognized_tracks[track_id]
            
            # Also clean from recognition cache
            if track_id in self.track_recognition_cache:
                del self.track_recognition_cache[track_id]
                print(f"[CLEANUP] Removed track {track_id} (cached person cleared)")
            else:
                print(f"[CLEANUP] Removed track {track_id}")

    # -------------------------------------------------------------

    def get_cache_stats(self):
        """
        Returns cache performance statistics.
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_lookups": total,
            "hit_rate_percent": hit_rate,
            "active_cached_tracks": len(self.track_recognition_cache)
        }

    # -------------------------------------------------------------
