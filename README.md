# YOLOv8 Object Detection with Real-Time Speech Feedback

Welcome to the **YOLOv8 Object Detection** project!  
This repository demonstrates how to use the powerful [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for real-time object detection on videos, enhanced with live speech feedback for detected objects.  
Whether you're a computer vision enthusiast, a developer, or just curious about AI, this project provides an interactive and impressive showcase of modern deep learning in action.

---

## 🚀 Features

- **Real-time object detection** on video files using YOLOv8.
- **Text-to-speech feedback**: Detected object classes are spoken aloud.
- **Easy customization**: Swap in your own videos for instant results.
- **Colorful bounding boxes** and class labels for clear visualization.

---

## 🛠️ Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/CodeXPandaa/yolov8-object-detection.git
```

### 2. Install Requirements

Make sure you have Python 3.8+ installed. Then, install the dependencies:

```sh
pip install -r requirements.txt
```

### 3. Download or Place a Sample Video

Place your sample video in the `inference/videos/` directory.  
For example, if your video is named `sample.mp4`, copy it to:

```
inference/videos/sample.mp4
```

### 4. Edit the Video Path

Open `yolov8_n_opencv.py` and **change line 34** to use your video file:

```python
cap = cv2.VideoCapture("inference/videos/sample.mp4")
```

### 5. Run the Object Detection Script

```sh
python yolov8_n_opencv.py
```

The script will process the video, display detected objects with bounding boxes, and announce detected classes via your speakers.

---

## 📂 Project Structure

- `yolov8_n_opencv.py` — Main script for video detection and speech.
- `utils/coco.txt` — List of COCO dataset class names.
- `weights/` — Pretrained YOLOv8 model weights.
- `inference/videos/` — Directory for your test videos.

---

## 🤖 Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

---

Enjoy exploring the world of AI-powered object detection!