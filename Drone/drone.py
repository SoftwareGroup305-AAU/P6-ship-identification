import olympe
import os
import time
from dataclasses import dataclass
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

#All caps indicate const
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")


@dataclass
class Location:    
    lat: float
    lon: float

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            lat=float(json_data['lat']),
            lon=float(json_data['lon'])
        )

def gps_callback(event, scheduler):
    print(f"Latitude: {event.args['latitude']}, "
          f"Longitude: {event.args['longitude']}, "
          f"Altitude: {event.args['altitude']}")

def test_takeoff():
    drone = olympe.Drone(DRONE_IP)
    drone.connect()
    listener = drone.subscribe(PositionChanged(), gps_callback)

    assert drone(TakeOff()).wait().success()
    time.sleep(10)
    assert drone(Landing()).wait().success()
    drone.disconnect()

if __name__ == "__main__":
    pass


# def main():
#     drone = olympe.Drone("192.168.42.1")
#     drone.connect()

#     listener = drone.subscribe(PositionChanged(), gps_callback)

#     cap = cv2.VideoCapture(f"rtsp://192.168.42.1/live")

#     try:
#         start_time = time.time()
#         while time.time() - start_time < 10:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Frame not received")
#                 break

#             cv2.imshow("Drone Stream", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         drone.unsubscribe(listener)
#         drone.disconnect()