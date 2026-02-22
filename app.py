import time
import threading
import cv2
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Shared state with main.py
output_frame = None
lock = threading.Lock()
final_results = {}
stop_signal = False

def generate():
    """Converts the latest processed frame into a JPEG stream for FastAPI"""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is not None:
                # Encode the frame as JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
                if flag:
                    # Yield the output frame as JPEG
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
        # Sleep for 30ms outside the lock to let the AI workers use the CPU
        time.sleep(0.03)

@app.get("/")
def index(request: Request):
    """FastAPI route for the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    """FastAPI route to serve the video feed"""
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_stream")
def video_stream_page(request: Request):
    """Renders the HTML page that embeds the video and the stop button."""
    return templates.TemplateResponse("stream.html", {"request": request})

@app.get("/stop_system")
def stop_trigger(request: Request):
    """Signal to stop the AI processing loop"""
    global stop_signal
    stop_signal = True
    return templates.TemplateResponse("results.html", {"request": request, "results_data": final_results})

@app.get("/results")
def results(request: Request):
    """Display final system statistics"""
    global final_results
    return templates.TemplateResponse("results.html", {"request": request, "results_data": final_results})

@app.post("/shutdown_server")
def shutdown_server():
    """Gracefully shutdown the FastAPI server"""
    def shutdown():
        time.sleep(1)
        os._exit(0)
    
    thread = threading.Thread(target=shutdown, daemon=True)
    thread.start()
    return {"status": "Server shutting down"}