from pathlib import Path
import random
import sys
import time
import ctypes
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue


def get_screen_resolution():
    """Retrieve primary monitor resolution with DPI awareness on Windows."""
    width, height = 1920, 1080
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
        except Exception:
            pass
    return width, height


# Project root directory (directory where this script is located)
BASE_DIR = Path(__file__).resolve().parent

# Load class list from utils/coco.txt (resolved relative to script directory)
coco_file = BASE_DIR / "utils" / "coco.txt"
if coco_file.is_file():
    with open(coco_file, "r") as my_file:
        data = my_file.read()
    class_list = [line.strip() for line in data.splitlines() if line.strip()]
else:
    class_list = []

# Load a pretrained YOLOv8n model
weights_path = BASE_DIR / "weights" / "yolov8n.pt"
model = YOLO(str(weights_path) if weights_path.is_file() else "yolov8n.pt", "v8")

# Fallback to model's built-in class names if coco.txt wasn't found
if not class_list and hasattr(model, "names") and model.names:
    class_list = [model.names[i] for i in range(len(model.names))]

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480


#  ----------------------------------------------------------
# Video file path (resolved relative to script directory)
video_path = BASE_DIR / "inference" / "videos" / "test 1.mp4"
#  ----------------------------------------------------------

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(str(video_path))

# Initialize text-to-speech queue and thread
speech_queue = queue.Queue()
stop_tts = threading.Event()

def tts_worker():
    # Initialize Windows COM for this thread
    if sys.platform == "win32":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

    # Use native Windows SAPI voice directly for rock-solid continuous speech without pyttsx3 loop death
    sapi_voice = None
    if sys.platform == "win32":
        try:
            import win32com.client
            sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            sapi_voice.Rate = 2  # Crisp, slightly faster rate for real-time video feedback
        except Exception:
            sapi_voice = None

    spoken_in_tts = set()

    while not stop_tts.is_set():
        try:
            text = speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if text is None or stop_tts.is_set():
            break

        # Collect any immediately pending words to speak together naturally without duplicate words
        words = []
        if text not in spoken_in_tts:
            words.append(text)
            spoken_in_tts.add(text)

        while not speech_queue.empty() and len(words) < 4:
            try:
                next_word = speech_queue.get_nowait()
                if next_word is None:
                    stop_tts.set()
                    break
                if next_word not in words and next_word not in spoken_in_tts:
                    words.append(next_word)
                    spoken_in_tts.add(next_word)
                speech_queue.task_done()
            except queue.Empty:
                break

        if not words:
            speech_queue.task_done()
            continue

        phrase = ", ".join(words)
        try:
            if sapi_voice is not None:
                sapi_voice.Speak(phrase)
            else:
                # Fallback to fresh pyttsx3 instance per speech call to prevent event loop death
                engine = pyttsx3.init()
                engine.setProperty("rate", 210)
                engine.say(phrase)
                engine.runAndWait()
                engine.stop()
                del engine
        except Exception:
            try:
                import win32com.client
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 2
                sapi_voice.Speak(phrase)
            except Exception:
                pass

        speech_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

if not cap.isOpened():
    print(f"Cannot open video source: {video_path}")
    exit()

# Native video frame rate for exact real-time playback
video_fps = cap.get(cv2.CAP_PROP_FPS)
if not video_fps or video_fps <= 0 or np.isnan(video_fps) or video_fps > 120:
    video_fps = 30.0
frame_duration = 1.0 / video_fps

# Shared state for background YOLO detection (keeps video at 100% native speed)
detection_lock = threading.Lock()
pending_frame = None
latest_detections = []  # List of tuples: (bb, clsID, conf, class_name)
new_frame_event = threading.Event()
stop_detection = threading.Event()

# Track already spoken classes so no duplicate words are ever spoken
already_spoken = set()

def detection_worker():
    global pending_frame, latest_detections
    while not stop_detection.is_set():
        if not new_frame_event.wait(timeout=0.1):
            continue
        new_frame_event.clear()

        with detection_lock:
            if pending_frame is None:
                continue
            frame_to_process = pending_frame
            pending_frame = None

        try:
            results = model.predict(
                source=[frame_to_process],
                conf=0.35,
                imgsz=640,
                verbose=False,
                save=False,
            )

            current_frame_classes = set()
            new_detections = []
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    clsID = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    bb = box.xyxy.cpu().numpy()[0]
                    class_name = class_list[clsID] if clsID < len(class_list) else str(clsID)
                    new_detections.append((bb, clsID, conf, class_name))
                    current_frame_classes.add(class_name)

            # Speak each detected object class once as it appears without any duplicate words
            for class_name in current_frame_classes:
                if class_name not in already_spoken:
                    already_spoken.add(class_name)
                    speech_queue.put(class_name)

            with detection_lock:
                latest_detections = new_detections
        except Exception:
            pass

# Start detection thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

window_name = "ObjectDetection"
# Use WINDOW_NORMAL so the window size is not constrained by video resolution
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Set default window size to full screen
screen_w, screen_h = get_screen_resolution()
cv2.resizeWindow(window_name, screen_w, screen_h)

# Keep aspect ratio for video content if supported by backend
if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_KEEPRATIO"):
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    except Exception:
        pass

hwnd = None
if sys.platform == "win32":
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    except Exception:
        hwnd = None

window_maximized = False

try:
    while True:
        frame_start = time.perf_counter()

        # Handle Windows window states (minimize, restore, close)
        if sys.platform == "win32":
            if not hwnd:
                try:
                    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                except Exception:
                    pass

            if hwnd:
                try:
                    # If window was destroyed / closed
                    if not ctypes.windll.user32.IsWindow(hwnd):
                        break

                    # If window is minimized, wait briefly and pause video rendering
                    if ctypes.windll.user32.IsIconic(hwnd):
                        key = cv2.waitKey(30) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            break
                        continue
                except Exception:
                    pass

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Send current frame to background detector if ready
        if pending_frame is None and not stop_detection.is_set():
            with detection_lock:
                pending_frame = frame.copy()
            new_frame_event.set()

        # Get latest detections to draw onto the frame
        with detection_lock:
            current_boxes = list(latest_detections)

        # Draw detected bounding boxes and class labels
        font = cv2.FONT_HERSHEY_COMPLEX
        for bb, clsID, conf, class_name in current_boxes:
            color = detection_colors[clsID % len(detection_colors)]
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                color,
                3,
            )
            label = f"{class_name} {round(conf, 2)}%"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), max(25, int(bb[1]) - 10)),
                font,
                0.9,
                (255, 255, 255),
                2,
            )

        # Display the resulting frame
        cv2.imshow(window_name, frame)

        # Maximize the window once upon first frame display
        if not window_maximized:
            if sys.platform == "win32":
                if not hwnd:
                    try:
                        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                    except Exception:
                        pass
                if hwnd:
                    try:
                        # SW_MAXIMIZE = 3: maximizes window while keeping title bar & minimize button
                        ctypes.windll.user32.ShowWindow(hwnd, 3)
                    except Exception:
                        pass
            window_maximized = True

        # Synchronize display to match the original video speed precisely
        elapsed = time.perf_counter() - frame_start
        wait_ms = max(1, int((frame_duration - elapsed) * 1000))

        # Process GUI events and key press
        key = cv2.waitKey(wait_ms) & 0xFF
        # Terminate run when 'q', 'Q', or ESC is pressed
        if key in (ord("q"), ord("Q"), 27):
            break

        # Terminate run when window close button (X) is clicked
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Stop background detection worker
    stop_detection.set()
    new_frame_event.set()
    detection_thread.join(timeout=1.0)

    # Cleanup: stop TTS thread promptly
    stop_tts.set()
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break
    speech_queue.put(None)
    tts_thread.join(timeout=1.0)

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
