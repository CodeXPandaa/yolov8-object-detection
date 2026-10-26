from pathlib import Path
import random
import sys
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

# Video file path (resolved relative to script directory)
video_path = BASE_DIR / "inference" / "videos" / "test0.mp4"
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(str(video_path))

# Initialize text-to-speech engine and queue
engine = pyttsx3.init()
speech_queue = queue.Queue()
stop_tts = threading.Event()

def tts_worker():
    while not stop_tts.is_set():
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None or stop_tts.is_set():
            break  # Exit signal
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

if not cap.isOpened():
    print(f"Cannot open video source: {video_path}")
    exit()

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
already_spoken = set()  # To avoid repeating the same class

try:
    while True:
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
        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #  resize the frame | small frame optimise the run
        # frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Predict on image
        detect_params = model.predict(source=[frame], conf=0.45, save=False)

        # Convert tensor array to numpy
        DP = detect_params[0].numpy()
        print(DP)

        if len(DP) != 0:
            spoken_classes = set()
            for i in range(len(detect_params[0])):
                print(i)

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                class_name = class_list[int(clsID)]
                cv2.putText(
                    frame,
                    class_name + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
                if class_name not in already_spoken:
                    speech_queue.put(class_name)
                    already_spoken.add(class_name)
                spoken_classes.add(class_name)

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

        # Process GUI events and key press
        key = cv2.waitKey(1) & 0xFF
        # Terminate run when 'q', 'Q', or ESC is pressed
        if key == ord("q") or key == ord("Q") or key == 27:
            break

        # Terminate run when window close button (X) is clicked
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Cleanup: stop TTS thread promptly
    stop_tts.set()
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break
    speech_queue.put(None)
    try:
        engine.stop()
    except Exception:
        pass
    tts_thread.join(timeout=1.0)

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
