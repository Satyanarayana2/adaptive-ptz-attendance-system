"""
⚡ PERFORMANCE PROFILER - Real-time System Monitoring
Automatically saves metrics per class session OR entire runtime if no session
Filename format: Session_based OR From_HH-MM-SS_to_HH-MM-SS_results.json
"""

import time
import threading
import json
import os
from collections import deque
from datetime import datetime


class PerformanceProfiler:
    """
    Real-time performance monitoring for the attendance system.
    - Saves metrics per class session (batch-section)
    - If no session running: saves entire runtime as From_HH-MM-SS_to_HH-MM-SS_results.json
    """

    def __init__(self, history_size=300):
        self.history_size = history_size
        self.current_session = None  # batch-section
        self.session_start_time = None
        self.system_start_time = time.time()
        self.system_start_datetime = datetime.now()
        
        # ===== FRAME TIMING =====
        self.camera_frame_times = deque(maxlen=history_size)
        self.process_frame_times = deque(maxlen=history_size)
        
        # ===== DETECTION & TRACKING =====
        self.detected_faces = deque(maxlen=history_size)
        self.tracked_faces = deque(maxlen=history_size)
        
        # ===== RECOGNITION =====
        self.recognition_results = deque(maxlen=history_size)
        
        # ===== QUALITY METRICS =====
        self.quality_scores = deque(maxlen=history_size)
        
        # ===== BUFFER METRICS =====
        self.buffer_metrics = deque(maxlen=history_size)
        
        # ===== COUNTERS =====
        self.total_frames_captured = 0
        self.total_frames_processed = 0
        self.total_faces_detected = 0
        self.total_faces_tracked = 0
        self.total_recognized = 0
        self.total_unknown = 0
        self.total_retried = 0
        self.total_instant_passes = 0
        
        # ===== TIMESTAMPS =====
        self.last_camera_frame_time = None
        self.last_process_start_time = None
        
        # ===== THREADING =====
        self.lock = threading.Lock()
        
        # ===== SESSION STORAGE =====
        self.sessions_data = {}  # batch-section -> report
        self.module_timings = {}
        self.successfully_recognized_tracks = {}
        
        print(f"\n[PROFILER] System initialized at {self.system_start_datetime.strftime('%H:%M:%S')}")

    def record_module_time(self, module_name, elapsed):
        with self.lock:
            if module_name not in self.module_timings:
                self.module_timings[module_name] = {"count": 0, "sum_ms": 0.0, "min_ms": float('inf'), "max_ms": 0.0}
            
            stats = self.module_timings[module_name]
            elapsed_ms = elapsed * 1000.0
            
            stats["count"] += 1
            stats["sum_ms"] += elapsed_ms
            if elapsed_ms < stats["min_ms"]: stats["min_ms"] = elapsed_ms
            if elapsed_ms > stats["max_ms"]: stats["max_ms"] = elapsed_ms
    
    # actual processing throughput - frames /  elaapsed time
    def _calculate_processing_fps(self):
        """actual processing throughput - frames /  elaapsed time"""
        elapsed = time.time() - self.system_start_time if self.session_start_time else 0
        with self.lock:
            return self.total_frames_processed / elapsed if elapsed > 0 else 0.0
        
    def _calculate_camera_fps(self):
        elapsed = time.time() - self.session_start_time if self.session_start_time else 0
        with self.lock:
            return self.total_frames_captured / elapsed if elapsed > 0 else 0.0
        
    def  _module_stats(self):
        stats = {}
        for mod, data in self.module_timings.items():
            if data["count"] > 0:
                stats[mod] = {
                    "count": data["count"],
                    "avg_ms": data["sum_ms"] / data["count"],
                    "min_ms": data["min_ms"],
                    "max_ms": data["max_ms"],
                }
        return stats
    # ===== SESSION MANAGEMENT =====
    
    def set_session(self, batch, section):
        """Called when session changes (from SessionController)"""
        # Save previous session if exists
        if self.current_session and self.session_start_time:
            self._save_session_report(self.current_session)
        
        # Start new session
        with self.lock:
            self.current_session = f"{batch}-{section}"
            self.session_start_time = time.time()
            
            # Reset counters for new session
            self.total_frames_captured = 0
            self.total_frames_processed = 0
            self.total_faces_detected = 0
            self.total_faces_tracked = 0
            self.total_recognized = 0
            self.total_unknown = 0
            self.total_retried = 0
            self.total_instant_passes = 0
            
            # Clear history for new session
            self.camera_frame_times.clear()
            self.process_frame_times.clear()
            self.quality_scores.clear()
            self.buffer_metrics.clear()
            self.module_timings.clear()
            self.successfully_recognized_tracks.clear()
            
            print(f"\n[PROFILER] Session started: {self.current_session}")

    def end_session(self):
        """Called when session ends (from SessionController)"""
        if self.current_session and self.session_start_time:
            self._save_session_report(self.current_session)
            print(f"[PROFILER] Session ended: {self.current_session}")
            self.current_session = None
            self.session_start_time = None

    def end_system(self):
        """Called when system shuts down (from main.py)"""
        # If there's an active session, save it
        if self.current_session and self.session_start_time:
            self._save_session_report(self.current_session)
        
        # Also save overall system runtime
        system_end_datetime = datetime.now()
        self._save_system_runtime(system_end_datetime)

    def _write_json_async(self, filename, report):
        def _write():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            try:
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"[PROFILER] Report saved asynchronously: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to save performance report: {e}")
                
        # Changed to daemon=False to ensure Python waits for the JSON write to finish during shutdown without corrupting the file
        threading.Thread(target=_write, daemon=False).start()

    def _save_session_report(self, session_id):
        """Save session report to disk"""
        report = self._generate_session_report(session_id)
        
        # Filename: batch-section_YYYY-MM-DD_HHMMSS.json
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"logs/performance/{session_id}_{timestamp_str}.json"
        
        # Also store in memory for quick access
        self.sessions_data[session_id] = report
        self._write_json_async(filename, report)

    def _save_system_runtime(self, end_datetime):
        """Save overall system runtime (when no session is running)"""
        report = self._generate_system_report(end_datetime)
        
        # Filename: YYYY-MM-DD_From_HH-MM_to_HH-MM.json
        date_str = self.system_start_datetime.strftime("%Y-%m-%d")
        start_time_str = self.system_start_datetime.strftime("%H-%M")
        end_time_str = end_datetime.strftime("%H-%M")
        filename = f"logs/performance/{date_str}_From_{start_time_str}_to_{end_time_str}.json"
        
        self._write_json_async(filename, report)

    def _generate_session_report(self, session_id):
        """Generate comprehensive report for a session"""
        elapsed_seconds = time.time() - self.session_start_time if self.session_start_time else 0
        
        report = {
            "type": "CLASS_SESSION",
            "session_id": session_id,
            "start_time": self.system_start_datetime.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": elapsed_seconds,
            
            # ===== FPS & TIMING =====
            "fps": {
                "camera_fps": self._calculate_camera_fps(),
                "processing_fps": self._calculate_processing_fps(),
                "avg_processing_time_ms": self._calculate_avg_processing_time(),
            },
            
            # ===== FRAME COUNTERS =====
            "frames": {
                "total_captured": self.total_frames_captured,
                "total_processed": self.total_frames_processed,
                "total_detected": self.total_faces_detected,
                "total_tracked": self.total_faces_tracked,
            },
            
            # ===== BUFFER STATISTICS =====
            "buffer_stats": {
                "instant_passes": self.total_instant_passes,
                "instant_pass_rate_percent": (self.total_instant_passes / max(self.total_frames_processed, 1) * 100),
            },
            
            # ===== QUALITY METRICS =====
            "quality_stats": self._calculate_quality_stats(),
            
            # ===== RECOGNITION RESULTS =====
            "recognition_stats": {
                "total_faces": self.total_recognized + self.total_unknown + self.total_retried,
                "recognized": self.total_recognized,
                "unknown": self.total_unknown,
                "retried": self.total_retried,
                "recognition_rate_percent": (self.total_recognized / max(self.total_recognized + self.total_unknown + self.total_retried, 1) * 100),
                "recognized_tracks": self.successfully_recognized_tracks
            },

            # ==== MODULE TIMINGS =====
            "module_stats": self._module_stats()
        }
        
        return report

    def _generate_system_report(self, end_datetime):
        """Generate report for entire system runtime (no session)"""
        elapsed_seconds = time.time() - self.system_start_time
        
        report = {
            "type": "SYSTEM_RUNTIME",
            "start_time": self.system_start_datetime.isoformat(),
            "end_time": end_datetime.isoformat(),
            "duration_seconds": elapsed_seconds,
            "session_info": "No lab session was running during this period",
            
            # ===== FPS & TIMING =====
            "fps": {
                "camera_fps": self._calculate_camera_fps(),
                "processing_fps": self._calculate_processing_fps(),
                "avg_processing_time_ms": self._calculate_avg_processing_time(),
            },
            
            # ===== FRAME COUNTERS =====
            "frames": {
                "total_captured": self.total_frames_captured,
                "total_processed": self.total_frames_processed,
                "total_detected": self.total_faces_detected,
                "total_tracked": self.total_faces_tracked,
            },
            
            # ===== BUFFER STATISTICS =====
            "buffer_stats": {
                "instant_passes": self.total_instant_passes,
                "instant_pass_rate_percent": (self.total_instant_passes / max(self.total_frames_processed, 1) * 100),
            },
            
            # ===== QUALITY METRICS =====
            "quality_stats": self._calculate_quality_stats(),
            
            # ===== RECOGNITION RESULTS =====
            "recognition_stats": {
                "total_faces": self.total_recognized + self.total_unknown + self.total_retried,
                "recognized": self.total_recognized,
                "unknown": self.total_unknown,
                "retried": self.total_retried,
                "recognition_rate_percent": (self.total_recognized / max(self.total_recognized + self.total_unknown + self.total_retried, 1) * 100),
                "recognized_tracks": self.successfully_recognized_tracks
            },

            # ==== MODULE TIMINGS =====
            "module_stats": self._module_stats()
        }
        
        return report


    # ===== FRAME TIMING =====
    
    def record_camera_frame(self):
        """Called when frame is read from camera"""
        now = time.time()
        
        with self.lock:
            if self.last_camera_frame_time is not None:
                frame_interval = now - self.last_camera_frame_time
                self.camera_frame_times.append(frame_interval)
            
            self.last_camera_frame_time = now
            self.total_frames_captured += 1

    def start_frame_processing(self):
        """Called at start of frame processing loop"""
        with self.lock:
            self.last_process_start_time = time.time()

    def end_frame_processing(self):
        """Called at end of frame processing loop"""
        if self.last_process_start_time is None:
            return
        
        elapsed = time.time() - self.last_process_start_time
        
        with self.lock:
            self.process_frame_times.append(elapsed)
            self.total_frames_processed += 1


    # ===== DETECTION & TRACKING =====
    
    def record_detection(self, face_count):
        """tracker.update() returns detected faces"""
        with self.lock:
            self.detected_faces.append(face_count)
            self.total_faces_detected += face_count

    def record_tracking(self, track_count):
        """Quality selector processes tracked faces"""
        with self.lock:
            self.tracked_faces.append(track_count)
            self.total_faces_tracked += track_count


    # ===== QUALITY SELECTION =====
    
    def record_quality_scores(self, sharpness, luminance, frontal):
        """QualitySelector calculates these scores"""
        with self.lock:
            self.quality_scores.append({
                "sharpness": sharpness,
                "luminance": luminance,
                "frontal": frontal,
                "timestamp": time.time()
            })

    def record_instant_pass(self):
        """QualitySelector.get_best() returns early"""
        with self.lock:
            self.total_instant_passes += 1
            self.buffer_metrics.append({
                "type": "instant_pass",
                "timestamp": time.time()
            })

    def record_buffer_usage(self, frames_used):
        """QualitySelector waited for min_frames"""
        with self.lock:
            self.buffer_metrics.append({
                "type": "buffered",
                "frames_used": frames_used,
                "timestamp": time.time()
            })


    # ===== RECOGNITION =====
    
    def record_recognition_result(self, track_id, matched, score=0.0, person_id=None):
        """Recognizer.recognize() returns result"""
        with self.lock:
            self.recognition_results.append({
                "track_id": track_id,
                "matched": matched,
                "score": score,
                "timestamp": time.time()
            })
            
            if matched:
                self.total_recognized += 1
                if person_id is not None:
                    self.successfully_recognized_tracks[str(track_id)] = person_id
            else:
                # Check if retry scenario (0.30-0.41)
                if 0.30 <= score <= 0.41:
                    self.total_retried += 1
                else:
                    self.total_unknown += 1


    # ===== CALCULATIONS =====
    
    def _calculate_camera_fps(self):
        """Calculate camera input FPS"""
        with self.lock:
            if len(self.camera_frame_times) == 0:
                return 0.0
            avg_interval = sum(self.camera_frame_times) / len(self.camera_frame_times)
            return 1.0 / avg_interval if avg_interval > 0 else 0.0

    def _calculate_processing_fps(self):
        """Calculate processing output FPS"""
        with self.lock:
            if len(self.process_frame_times) == 0:
                return 0.0
            avg_time = sum(self.process_frame_times) / len(self.process_frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0

    def _calculate_avg_processing_time(self):
        """Get average frame processing time in ms"""
        with self.lock:
            if len(self.process_frame_times) == 0:
                return 0.0
            return (sum(self.process_frame_times) / len(self.process_frame_times)) * 1000.0

    def _calculate_quality_stats(self):
        """Calculate average quality metrics"""
        with self.lock:
            if len(self.quality_scores) == 0:
                return {
                    "avg_sharpness": 0,
                    "avg_luminance": 0,
                    "avg_frontal": 0,
                }
            
            sharpness_vals = [q["sharpness"] for q in self.quality_scores]
            luminance_vals = [q["luminance"] for q in self.quality_scores]
            frontal_vals = [q["frontal"] for q in self.quality_scores]
            
            return {
                "avg_sharpness": sum(sharpness_vals) / len(sharpness_vals),
                "avg_luminance": sum(luminance_vals) / len(luminance_vals),
                "avg_frontal": sum(frontal_vals) / len(frontal_vals),
                "min_sharpness": min(sharpness_vals),
                "max_sharpness": max(sharpness_vals),
            }


    # ===== REPORTING =====
    
    def print_session_report(self, session_id=None):
        """Print formatted session report"""
        if session_id is None:
            session_id = self.current_session
        
        if session_id not in self.sessions_data:
            print(f"[ERROR] No report found for session: {session_id}")
            return
        
        report = self.sessions_data[session_id]
        self._print_report(report)

    def _print_report(self, report):
        """Print formatted report"""
        print("\n" + "="*80)
        print(f"PERFORMANCE REPORT - {report.get('type', 'UNKNOWN')}")
        if report.get('session_id'):
            print(f"Session: {report['session_id']}")
        print(f"Time: {report['start_time']} to {report['end_time']}")
        print("="*80)
        
        print(f"\n[TIME] DURATION: {report['duration_seconds']:.1f} seconds")
        
        print(f"\n[FPS METRICS]:")
        print(f"   Camera FPS:        {report['fps']['camera_fps']:.1f} fps")
        print(f"   Processing FPS:    {report['fps']['processing_fps']:.1f} fps")
        print(f"   Avg Process Time:  {report['fps']['avg_processing_time_ms']:.1f} ms/frame")
        
        print(f"\n[FRAME COUNTERS]:")
        print(f"   Total Captured:    {report['frames']['total_captured']}")
        print(f"   Total Processed:   {report['frames']['total_processed']}")
        print(f"   Total Detected:    {report['frames']['total_detected']}")
        print(f"   Total Tracked:     {report['frames']['total_tracked']}")
        
        print(f"\n[BUFFER STATISTICS]:")
        bs = report['buffer_stats']
        print(f"   Instant Passes:    {bs['instant_passes']} ({bs['instant_pass_rate_percent']:.1f}%)")
        
        print(f"\n[QUALITY METRICS]:")
        qs = report['quality_stats']
        print(f"   Avg Sharpness:     {qs['avg_sharpness']:.1f} σ²")
        print(f"   Sharpness Range:   {qs['min_sharpness']:.1f} - {qs['max_sharpness']:.1f}")
        print(f"   Avg Luminance:     {qs['avg_luminance']:.1f}")
        print(f"   Avg Frontal:       {qs['avg_frontal']:.3f}")
        
        print(f"\n[RECOGNITION RESULTS]:")
        rs = report['recognition_stats']
        print(f"   Total Faces:       {rs['total_faces']}")
        print(f"   Recognized:        {rs['recognized']} ({rs['recognition_rate_percent']:.1f}%)")
        print(f"   Unknown:           {rs['unknown']}")
        print(f"   Retried:           {rs['retried']}")
        
        print("\n" + "="*80 + "\n")
        if "module_stats" in report:
            print("\n [module timings]: avg ms per module")
            for mod, stats in report["module_stats"].items():
                print(f"    {mod}: count={stats['count']}, avg={stats['avg_ms']:.1f}ms, min={stats['min_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")
    

    def get_all_sessions_summary(self):
        """Get summary of all sessions in memory"""
        return self.sessions_data