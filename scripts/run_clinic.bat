@echo off
setlocal

REM Configure your RTSP URL here
set RTSP_URL=rtsp://admin:PASS@CAM_IP:554/cam/realmonitor?channel=1&subtype=0

cd /d "%~dp0.."
call .\.venv\Scripts\activate
python main.py --source "%RTSP_URL%" --reid --no-window

endlocal


