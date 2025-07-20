import random
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)
# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("inference/videos/afriq1.MP4")

# Initialize text-to-speech engine and queue
engine = pyttsx3.init()
speech_queue = queue.Queue()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break  # Exit signal
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

already_spoken = set()  # To avoid repeating the same class

while True:
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
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup: stop TTS thread
speech_queue.put(None)
tts_thread.join()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
