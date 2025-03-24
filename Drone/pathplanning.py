import requests
import asyncio
from dataclasses import dataclass
import math
from urllib.parse import quote
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import json
import osmnx as ox
from pyproj import CRS, Transformer

SECTOR_SIZE = 500 #subgrid size in meters

def extract_prot_area(wanted_area):
    protected_area_name = wanted_area
    protected_area = ox.geocode_to_gdf(protected_area_name)#osm lib data extractor/wrapper
    geometry = protected_area.geometry.iloc[0]

    if isinstance(geometry, Polygon):
        return list(geometry.exterior.coords)
    elif isinstance(geometry, MultiPolygon):
        largest_polygon = max(geometry.geoms, key=lambda p: p.area)
        return list(largest_polygon.exterior.coords)
    else:
        raise TypeError("oh no, extraction failed :(")


async def fetch_area_bb(query): 
  headers = {'User-Agent':'DroneMap/1.0 (mail+osm@mail.dk)'}
  response = requests.get(f'https://nominatim.openstreetmap.org/search?q=${quote(query)}&format=json', headers=headers)#quote simply transforms our search param into a URI compatible string (quote('abc def') -> 'abc%20def')
  
  if (response.ok):
    data = json.loads(response.content.decode('utf-8'))
    length = len(data)
    return data[length-1]["boundingbox"]
  
  #print(response._content)


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


# Visual Grid creation (I hate )
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
    
    #create our sectors (grid (our 500x500meter gris of the area)) in UTM
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

async def main():
    area = "Nibe-Gjøl Bredning Vildtreservat"
    bb_coords = await fetch_area_bb(area)
    protected_area = extract_prot_area(area)
    lat1, lon1 = float(bb_coords[0]), float(bb_coords[2])
    lat2, lon2 = float(bb_coords[1]), float(bb_coords[3])
    grid_gdf = create_geographic_grid(lat1, lon1, lat2, lon2, sector_size_m=SECTOR_SIZE)

    #For visualisation
    show_grid(protected_area, grid_gdf, area)

if __name__ == "__main__":
    asyncio.run(main())
