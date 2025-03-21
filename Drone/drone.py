import olympe
import cv2
import time
from olympe.messages.ardrone3.PilotingState import PositionChanged

def gps_callback(event, scheduler):
    print(f"Latitude: {event.args['latitude']}, "
          f"Longitude: {event.args['longitude']}, "
          f"Altitude: {event.args['altitude']}")

def main():
    drone = olympe.Drone("192.168.42.1")
    drone.connect()

    listener = drone.subscribe(PositionChanged(), gps_callback)

    cap = cv2.VideoCapture(f"rtsp://192.168.42.1/live")

    try:
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                print("Frame not received")
                break

            cv2.imshow("Drone Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        drone.unsubscribe(listener)
        drone.disconnect()

if __name__ == "__main__":
    main()