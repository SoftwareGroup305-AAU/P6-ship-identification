import olympe
import os
import time
import requests
import asyncio
from dataclasses import dataclass
from urllib.parse import quote
from olympe.messages.ardrone3.Piloting import TakeOff, Landing

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

async def fetch_gps_location(query): 
  headers = {'User-Agent':'DroneMap/1.0 (mail+osm@mail.dk)'}
  response = requests.get(f'https://nominatim.openstreetmap.org/search?q=${quote(query)}&format=json', headers=headers)
  print(response._content)
  print("Reply^")

  #if (response.statusCode == 200) {
  #  final List<dynamic> data = json.decode(response.body);
  #  return data.map((item) => Location.fromJson(item)).toList();
  #} else {
  #  throw Exception('Failed to load location data');
  #}
#}

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


async def main():
    await fetch_gps_location("doggerbank")

if __name__ == "__main__":
    asyncio.run(main())


# import olympe
# import cv2
# import time
# from olympe.messages.ardrone3.PilotingState import PositionChanged

# def gps_callback(event, scheduler):
#     print(f"Latitude: {event.args['latitude']}, "
#           f"Longitude: {event.args['longitude']}, "
#           f"Altitude: {event.args['altitude']}")

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

# if __name__ == "__main__":
#     main()