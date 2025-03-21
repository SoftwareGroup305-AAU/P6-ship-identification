import argparse
import olympe
import os
import re
import sys
import cv2
import time
from olympe.messages.onboard_tracker import start_tracking_engine

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")


def main(argv):
    parser = argparse.ArgumentParser(description="Olympe OpenCV Streaming Example")
    parser.add_argument(
        "-u",
        "--url",
        default=f"rtsp://{DRONE_IP}:{DRONE_RTSP_PORT}/live",
        help="RTSP stream URL (default: Parrot drone live stream)",
    )

    args = parser.parse_args(argv)

    drone_ip = re.search(r"\d+\.\d+\.\d+\.\d+", args.url)
    drone = olympe.Drone(drone_ip.group())
    drone.connect()

    drone(start_tracking_engine(box_proposals=True)).wait()

    cap = cv2.VideoCapture(args.url)

    if not cap.isOpened():
        print("Error: Cannot open video stream")
        drone.disconnect()
        return

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame")
                break

            cv2.imshow('Drone Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Streaming for 10 seconds (similar to your original script)
            if time.time() - start_time > 10:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        drone.disconnect()


def test_stream():
    main([])


if __name__ == "__main__":
    main(sys.argv[1:])