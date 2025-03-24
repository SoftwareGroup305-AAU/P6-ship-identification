import olympe
import os
import time
from dataclasses import dataclass
import math
from urllib.parse import quote
from pyproj import CRS, Transformer
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import json
import osmnx as ox

#All caps indicate const
DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
SECTOR_SIZE = 500 #subgrid size in meters


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

def extract_prot_area(wanted_area):
    protected_area_name = wanted_area
    protected_area = ox.geocode_to_gdf(protected_area_name)#osm lib data extractor/wrapper
    geometry = protected_area.geometry.iloc[0]

    print(geometry.bounds)
    data = {
        "bounds": geometry.bounds
        #"protected": list.clear()
    }
    if isinstance(geometry, Polygon):
        data.update({"protected": list(geometry.exterior.coords)})
    elif isinstance(geometry, MultiPolygon):
        largest_polygon = max(geometry.geoms, key=lambda p: p.area)
        data.update({"protected": list(largest_polygon.exterior.coords)})
    else:
        raise TypeError("oh no, extraction failed :(")
    
    return data

def show_grid(protected_area, bb, area):
    polygon = Polygon(protected_area)

    # Combined Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot protected area
    x, y = polygon.exterior.xy
    
    # Protected area
    ax.fill(x, y, alpha=1, fc='green', label='Protected Area')

    ax.plot(x, y, color='green', linewidth=2)
    
    # Grid cells from GeoDataFrame
    bb.plot(ax=ax, facecolor='red', edgecolor='black', alpha=0.2, linewidth=0.5, label='Areas to ignore')#Not sure why the label isn't appearing
    
    ax.scatter(x, y, color='blue', s=10, label='Border gps locations')
    ax.set_title(f'Protected Area Visual ({area})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(False)
    ax.legend()
    
    plt.axis('equal')
    plt.savefig('combined_plot.png', transparent=True)
    plt.show()


# Visual Grid creation
def create_geographic_grid(lat1, lon1, lat2, lon2, sector_size_m):
    #identify WGS boundariers
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    #We find the center/midpoint lattitude and longtitude 
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    #UTM = Universal Transverse Mercator is a map projection method that allows us to 'transform' the lat+long into a coordinate system
    utm_zone = int((center_lon + 180) / 6) + 1
    #identify hemesphere
    is_northern = center_lat >= 0

    #We create a Coordinate Reference Systems, based on our utm zone
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'south': not is_northern})
    #We create two tranformers, one for wgs -> utm, an the inversee 
    transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    transformer_to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    #identify utm boundariers
    utm_minx, utm_miny = transformer_to_utm.transform(min_lon, min_lat)
    utm_maxx, utm_maxy = transformer_to_utm.transform(max_lon, max_lat)
    
    #create our sectors (grid (our 500x500meter grid of the area)) in UTM
    polygons = []
    x = utm_minx
    while x < utm_maxx:
        y = utm_miny
        while y < utm_maxy:
            polygon = Polygon([
                (x, y),
                (x + sector_size_m, y),
                (x + sector_size_m, y + sector_size_m),
                (x, y + sector_size_m)
            ])
            polygons.append(polygon)
            y += sector_size_m
        x += sector_size_m
    geo_polygons = [Polygon([transformer_to_wgs.transform(x, y) for x, y in poly.exterior.coords]) for poly in polygons]#convert sectors to WGS
    return gpd.GeoDataFrame(geometry=geo_polygons, crs="EPSG:4326")#return our grid

def calculate_grid(lat1, lon1, lat2, lon2):#I cooked this before i got distracted by creating the visual grid, and before i got the proper data from OSM, so we might not need it. But now we have it
    """Uses the haversine formula to calculate the great-circle distance between two points, aka
    the shortest distance over the earth’s surface assuming no mountain is in the way: https://en.wikipedia.org/wiki/Haversine_formula"""
    
    R = 6371e3  # Earth radius in meters

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # distance in meters

    print(f"distance: {distance}")
    delta_lat = math.radians(abs(lat2 - lat1))
    delta_lon = math.radians(abs(lon2 - lon1))

    # Average latitude for longitude correction
    avg_lat = math.radians((lat1 + lat2) / 2)

    # Calculate North-South distance
    ns_dist = R * delta_lat

    # Calculate East-West distance (corrected by latitude)
    ew_dist = R * delta_lon * math.cos(avg_lat)

    # Area calculation (rectangle approximation)
    area_m2 = ns_dist * ew_dist  # area in square meters
    area_km2 = area_m2 / 1e6     # convert to square kilometers
    
    print(f"Total area: {area_km2:.2f} km²")
    subarea = area_km2 / 4
    print(f"Sub area: {subarea} km²")
    # Grid squares of 500m x 500m

def main():
    area = "Nibe-Gjøl Bredning Vildtreservat"
    data = extract_prot_area(area)
    bb_coords = data['bounds']
    protected_area = data['protected']
    lat1, lon1 = float(bb_coords[1]), float(bb_coords[0])
    lat2, lon2 = float(bb_coords[3]), float(bb_coords[2])
    grid_gdf = create_geographic_grid(lat1, lon1, lat2, lon2, sector_size_m=SECTOR_SIZE)

    show_grid(protected_area, grid_gdf, area)

if __name__ == "__main__":
    main()


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