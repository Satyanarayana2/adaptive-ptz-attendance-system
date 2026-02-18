import time
from datetime import datetime

class SessionController:
    """
    The State Manager (Brain).
    Holds 'Current' and 'Next' session to manage transitions smoothly.
    """
    
    # PTZ Configuration
    ENTRY_BUFFER_MINUTES = 15  # First 15 mins: Watch Entrance
    SCAN_INTERVAL_SECONDS = 120 # Every 2 mins: Rotate Row View
    SCAN_PRESETS = ["ROW_1", "ROW_2", "ROW_3", "ROW_4", "CLASS_OVERVIEW"]

    def __init__(self, db, ptz, recognizer):
        self.db = db
        self.ptz = ptz
        self.recognizer = recognizer
        
        # --- THE STATE MEMORY ---
        self.current_session = None
        self.next_session = None
        self.state = "UNKNOWN" # IDLE, ENTRY, SCAN
        
        # Timers
        self.scan_index = 0
        self.last_move_time = 0

    def update(self):
        """
        Called every frame. Refreshes state and decides action.
        """
        # 1. Refresh Knowledge (Get Current & Next)
        schedule = self.db.get_scheduler_state()
        new_current = schedule['current']
        self.next_session = schedule['next'] # Store for future logic (e.g. countdowns)

        # 2. Detect State Change
        # Case A: We entered a NEW class
        if new_current and (self.current_session is None or new_current['class_id'] != self.current_session['class_id']):
            print(f"[CONTROLLER] Session Started: {new_current['batch']}")
            self.transition_to_class(new_current)

        # Case B: We exited a class (Current is None, but we had one before)
        elif new_current is None and self.current_session is not None:
            print("[CONTROLLER] Session Ended. Switching to IDLE.")
            self.transition_to_idle()

        # Update our memory
        self.current_session = new_current

        # 3. Execute Continuous Logic
        if self.current_session:
            self.handle_class_logic()
        else:
            self.handle_idle_logic()

    def transition_to_class(self, session):
        """Setup for a new class."""
        self.current_session = session
        
        # 1. Load Students for this class
        self.recognizer.reload_gallery(session['class_id'])
        
        # 2. Reset to Entry Mode
        self.state = "ENTRY"
        self.ptz.goto_preset("ENTRANCE_VIEW")
        print(f"[CONTROLLER] Mode: ENTRY (Next class is: {self.next_session['batch'] if self.next_session else 'None'})")

    def transition_to_idle(self):
        """Reset for break/end of day."""
        self.state = "IDLE"
        
        # 1. Unload Students (Security Mode)
        self.recognizer.reload_gallery(None)
        
        # 2. Watch Door
        self.ptz.goto_preset("ENTRANCE_VIEW")

        if self.next_session:
            start = self.next_session['start_time']
            print(f"[CONTROLLER] Idle Mode. Next class starts at {start}.")

    def handle_class_logic(self):
        """Decides: Watch Door (Entry) OR Scan Rows."""
        now = datetime.now()
        start_time = datetime.combine(now.date(), self.current_session['start_time'])
        elapsed_mins = (now - start_time).total_seconds() / 60

        # Phase 1: Entry (0-15 mins)
        if elapsed_mins <= self.ENTRY_BUFFER_MINUTES:
            if self.state != "ENTRY":
                self.state = "ENTRY"
                self.ptz.goto_preset("ENTRANCE_VIEW")
        
        # Phase 2: Row Scan (15 mins - End)
        else:
            if self.state != "SCAN":
                self.state = "SCAN"
                self.scan_index = 0
                self.last_move_time = 0 # Move immediately
            
            self._process_scan_rotation()

    def handle_idle_logic(self):
        """Logic for breaks (optional GrandTour here)."""
        # For now, just hold the entrance view
        pass

    def _process_scan_rotation(self):
        """Cycles through presets."""
        if not self.ptz: return
        now = time.time()
        
        if now - self.last_move_time > self.SCAN_INTERVAL_SECONDS:
            preset = self.SCAN_PRESETS[self.scan_index]
            print(f"[CONTROLLER]  Row Scan: Moving to {preset}")
            self.ptz.goto_preset(preset)
            
            self.last_move_time = now
            self.scan_index = (self.scan_index + 1) % len(self.SCAN_PRESETS)