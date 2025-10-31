import os
import sys
import cv2


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rtsp_probe.py <rtsp_url>")
        return 2

    # Improve RTSP reliability for OpenCV's FFmpeg backend
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000"

    url = sys.argv[1]
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    opened = cap.isOpened()
    if not opened:
        print("opened= False")
        return 1
    ret, _ = cap.read()
    cap.release()
    print(f"opened= True, first_read= {ret}")
    return 0 if ret else 1


if __name__ == "__main__":
    raise SystemExit(main())

