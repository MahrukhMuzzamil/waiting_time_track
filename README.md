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

- Headless (for autostart on Windows):
```powershell
python main.py --source "rtsp://user:pass@CAM_IP:554/..." --reid --no-window
```

- Enable ReID for more stable IDs across occlusions/exits:
```powershell
python main.py --show-fps --reid
```

Press `q` to quit the window.

## Options
- `--conf`: detection confidence threshold (default 0.4)
- `--max-missing`: how many frames to keep a track alive without detection (default 30)
- `--iou`: IoU threshold to match detections to tracks (default 0.3)
- `--show-fps`: overlay FPS counter
 - `--no-window`: run without display window (for background/service use)
 - `--reid`: enable ReID memory to persist identity across occlusions/exits
 - `--reid-sim`: cosine similarity threshold for ReID (default 0.62)

## Notes
- The model automatically downloads `yolov8n.pt` on first run.
- For best results, ensure good lighting and a clear view of people.
- This is a prototype with a simple tracker; in a clinic deployment, you can switch to a stronger tracker (e.g., ByteTrack/DeepSORT) and add patient identification logic.
 - ReID uses a pretrained ResNet18 embedding. It’s CPU-capable but slower than no-ReID; for better performance, use a GPU.

## Windows Autostart (Scheduled Task)
1. Edit `scripts\run_clinic.bat` and set your RTSP URL.
2. Register a Scheduled Task (runs at logon, restarts on failure):
```powershell
PowerShell -ExecutionPolicy Bypass -File scripts\register_autostart.ps1 -TaskName ClinicWaitTimeApp
```
3. To uninstall:
```powershell
Unregister-ScheduledTask -TaskName ClinicWaitTimeApp -Confirm:$false
```

## Guard Console (Local Deployment)
Run the vision stack on the guard’s PC so the stream stays on-prem and starts automatically.

1. Ensure `config.json` contains the camera RTSP url (`rtsp_url`) or pass `-RtspUrl` on the command.
2. Create the virtual environment and install dependencies (see setup above).
3. Launch the guard dashboard manually:
   ```powershell
   PowerShell -ExecutionPolicy Bypass -File scripts\start_guard_console.ps1 -RtspUrl "rtsp://user:pass@CAM_IP:554/..."
   ```
   The script boots `server.py` in a minimized PowerShell window and opens `http://localhost:8000/video_ai` in the default browser.
   - For a double-click experience, create a desktop shortcut that points to `scripts\launch_guard_console.bat`. Once the RTSP URL is saved in `config.json`, the guard only needs to open that shortcut.
4. Optional: register a scheduled task that starts the dashboard at logon (runs kiosk-style):
   ```powershell
   $project = "C:\Users\Guard\Desktop\ai_track_app"
   PowerShell -ExecutionPolicy Bypass -File "$project\scripts\register_autostart.ps1" `
     -TaskName GuardConsole `
     -Executable powershell.exe `
     -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$project\scripts\start_guard_console.ps1`""
   ```
5. To remove the scheduled task later:
   ```powershell
   Unregister-ScheduledTask -TaskName GuardConsole -Confirm:$false
   ```

## Deploying to Render
If you want the stream accessible off-site, deploy the Flask app to Render.

1. Push the repo to GitHub/GitLab and connect it to a new Render Web Service.
2. Choose the **Starter** plan (the free tier sleeps and doesn’t have enough CPU/RAM for the model load).
3. During setup:
   - Set `Build Command` to the value already defined in `render.yaml` (`pip install --upgrade pip && pip install -r requirements.txt`).
   - Set `Start Command` to `./scripts/render_start.sh`.
   - Add a secret environment variable `RTSP_URL` with the camera address.
   - Leave `REID` unset (defaults to `0` on Render to reduce load).
4. After the first deploy finishes, hit the `/snapshot` endpoint to confirm the camera connection, then switch to `/video_ai` for the annotated stream. Render proxies support the MJPEG feed used by the dashboard.
5. Keep an eye on the service metrics; if CPU stays pegged, consider moving to a larger instance or enabling ReID only on beefier hardware.
