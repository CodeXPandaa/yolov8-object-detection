from pathlib import Path
from ultralytics import YOLO
import numpy

BASE_DIR = Path(__file__).resolve().parent

# load a pretrained YOLOv8n model
weights_path = BASE_DIR / "weights" / "yolov8n.pt"
model = YOLO(str(weights_path) if weights_path.is_file() else "yolov8n.pt", "v8")  

# predict on an image
image_path = BASE_DIR / "inference" / "images" / "img0.JPG"
detection_output = model.predict(source=str(image_path), conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
