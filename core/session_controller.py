import time
from datetime import datetime
from utils.ptz.presets import ENTRANCE_VIEW, WIDE_VIEW, RIGHT_CORNER_VIEW

class SessionController:
    """
    The State Manager (Brain).
    Handles PTZ movement, toggles Adaptive Learning, and RESETS the Tracker on movement.
    """
    ENTRY_BUFFER_MINUTES = 15  # First 15 mins: Watch Entrance

    def __init__(self, db, ptz, adaptive_manager, tracker):
        self.db = db
        self.ptz = ptz
        self.adaptive_manager = adaptive_manager
        self.tracker = tracker 
        
        self.current_session = None
        self.next_session = None
        self.state = "UNKNOWN"

    def update(self):
        """Called every frame. Refreshes state and decides action."""
        schedule = self.db.get_scheduler_state()
        new_current = schedule['current']
        self.next_session = schedule['next']

        # Class just started (or an Overlap triggered a new class)
        if new_current and (self.current_session is None or new_current['class_id'] != self.current_session['class_id']):
            print(f"[CONTROLLER] Session Started: {new_current['batch']}-{new_current['section']}")
            self.current_session = new_current
            self.state = "UNKNOWN" # Force state re-evaluation

        # Class just ended
        elif new_current is None and (self.current_session is not None or self.state == "UNKNOWN"):
            print("[CONTROLLER] Session Ended. Switching to IDLE.")
            self.transition_to_idle()

        self.current_session = new_current

        if self.current_session:
            self.handle_class_logic()

    def transition_to_idle(self):
        """Reset for break/end of day."""
        self.current_session = None
        self._change_state("IDLE", ENTRANCE_VIEW, enable_learning=True)
        if self.next_session:
            print(f"[CONTROLLER] Idle Mode. Next class starts at {self.next_session['start_time']}.")

    def _change_state(self, new_state, preset_name, enable_learning):
        """Helper to safely change physical camera state and reset tracking"""
        if self.state != new_state:
            print(f"[CONTROLLER] Phase changed to {new_state}. Learning: {enable_learning}")
            self.state = new_state
            
            # 1. Toggle learning
            self.adaptive_manager.set_learning_mode(enable_learning)
            
            # 2. CLEAR THE TRACKER! (Prevents bounding boxes from jumping across the room)
            if self.tracker:
                self.tracker.tracks.clear()
                
            # 3. Move camera
            if self.ptz:
                self.ptz.goto_preset(preset_name)

    def handle_class_logic(self):
        """Decides: Watch Door (Entry) OR Alternate Scan (Wide/Corner)."""
        now = datetime.now()
        start_time = datetime.combine(now.date(), self.current_session['start_time'])
        
        elapsed_mins = (now - start_time).total_seconds() / 60.0

        # Phase 1: Entry (0-15 mins)
        if elapsed_mins <= self.ENTRY_BUFFER_MINUTES:
            self._change_state("ENTRY", ENTRANCE_VIEW, enable_learning=True)

        # Phase 2: Alternating Scan (15 mins -> End of Class)
        else:
            scan_time_mins = elapsed_mins - self.ENTRY_BUFFER_MINUTES
            
            # Modulo 10 gives us a repeating 10-minute cycle. 
            # 0 to 4.99 = Wide View. 5 to 9.99 = Corner View.
            if (scan_time_mins % 10) < 5:
                self._change_state("SCAN_WIDE", WIDE_VIEW, enable_learning=False)
            else:
                self._change_state("SCAN_CORNER", RIGHT_CORNER_VIEW, enable_learning=False)