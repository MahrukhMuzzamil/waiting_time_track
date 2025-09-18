# Clinic Wait-Time Prototype (Webcam)

This prototype detects people in real time from a webcam (or video) and overlays each person's waiting time above their head. It uses YOLO for person detection and a lightweight IoU-based tracker to keep a stable ID per person.

## Prerequisites
- Python 3.10 or 3.11 recommended
- Windows/Mac/Linux with a webcam or a video file

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
- Default webcam:
```powershell
python main.py --show-fps
```
- Specific camera index (e.g., 1) or a video file path:
```powershell
python main.py --source 1
python main.py --source path\to\video.mp4
```

Press `q` to quit the window.

## Options
- `--conf`: detection confidence threshold (default 0.4)
- `--max-missing`: how many frames to keep a track alive without detection (default 30)
- `--iou`: IoU threshold to match detections to tracks (default 0.3)
- `--show-fps`: overlay FPS counter

## Notes
- The model automatically downloads `yolov8n.pt` on first run.
- For best results, ensure good lighting and a clear view of people.
- This is a prototype with a simple tracker; in a clinic deployment, you can switch to a stronger tracker (e.g., ByteTrack/DeepSORT) and add patient identification logic.
