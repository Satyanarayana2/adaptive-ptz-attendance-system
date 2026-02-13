import time
from flask import Flask, Response, render_template, request
import threading
import cv2
import os

app = Flask(__name__)

# Shared state with main.py
output_frame = None
lock = threading.Lock()
final_results = {}
stop_signal = False

def generate():
    """Converts the latest processed frame into a JPEG stream for Flask"""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                pass
            else:
                # Encode the frame as JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
                if flag:
                    # Yield the output fame as JPEG
                    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+ bytearray(encodedImage) + b'\r\n')
            
        # Sleep for 30ms outside thr lock to let the AI workers use the CPU
        time.sleep(0.03)
            

@app.route("/")
def index():
    """Flask route for the main page"""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Flask route to serve the video feed"""
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_stream")
def video_stream_page():
    """Renders the HTML page that embeds the video and the stop button."""
    return render_template("stream.html")

@app.route("/stop_system")
def stop_trigger():
    """Signal to stop the AI processing loop"""
    global stop_signal
    stop_signal = True
    return render_template("results.html", results_data=final_results)

@app.route("/results")
def results():
    """Display final system statistics"""
    global final_results
    return render_template("results.html", results_data=final_results)

@app.route("/shutdown_server", methods=["POST"])
def shutdown_server():
    """Gracefully shutdown the Flask server"""
    def shutdown():
        import time
        time.sleep(1)
        os._exit(0)
    
    thread = threading.Thread(target=shutdown)
    thread.daemon = True
    thread.start()
    return {"status": "Server shutting down"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
