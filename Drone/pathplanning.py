from dataclasses import dataclass
import math
from urllib.parse import quote
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import json
import osmnx as ox
from pyproj import CRS, Transformer
import threading

SECTOR_SIZE = 500 #subgrid size in meters
MAXSIZE_TSP = float('inf')
N = 4
final_path = [None] * (N+1)
visited = [False] * N
final_res = MAXSIZE_TSP #final min weight of route

adj = [[0, 10, 15, 20],
       [10, 0, 35, 25],
       [15, 35, 0, 30],
       [20, 25, 30, 0]]


def first_min(adj, i):
    min = MAXSIZE_TSP
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]
 
    return min
 
# function to find the second minimum edge 
# cost having an end at the vertex i
def second_min(adj, i):
    first, second = MAXSIZE_TSP, MAXSIZE_TSP
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
 
        elif(adj[i][j] <= second and
             adj[i][j] != first):
            second = adj[i][j]
 
    return second

def TSP_Rec(adj, curr_bound, curr_weight, level, curr_path, visited):
    global final_res
     
    if level == N:
         
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
             
            curr_res = curr_weight + adj[curr_path[level - 1]]\
                                        [curr_path[0]]
            if curr_res < final_res:
                copy_final_path(curr_path)
                final_res = curr_res
        return
 
    for i in range(N):
         
        if (adj[curr_path[level-1]][i] != 0 and
                            visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]
 
            if level == 1:
                curr_bound -= ((first_min(adj, curr_path[level - 1]) +
                                first_min(adj, i)) / 2)
            else:
                curr_bound -= ((second_min(adj, curr_path[level - 1]) +
                                 first_min(adj, i)) / 2)
 
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True
                 
                # call TSPRec for the next level
                TSP_Rec(adj, curr_bound, curr_weight, level + 1, curr_path, visited)
 
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp
 
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

def TSP(adj):
    curr_bound = 0
    curr_path = [-1] * (N+1)
    visited = [False] * N

    for i in range(N):
        curr_bound += (first_min(adj, i) + second_min(adj, i))

    curr_bound = math.ceil(curr_bound / 2)

    visited[0] = True
    curr_path[0] = 0

    TSP_Rec(adj, curr_bound, 0, 1, curr_path, visited)

def copy_final_path(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]
    

def extract_prot_area(wanted_area):
    protected_area_name = wanted_area
    protected_area = ox.geocode_to_gdf(protected_area_name)#osm lib data extractor/wrapper
    geometry = protected_area.geometry.iloc[0]

    data = {
        "bounds": geometry.bounds
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

renderThread = None

def main():
    global renderThread
    TSP(adj)
 
    print("Minimum cost :", final_res)
    print("Path Taken : ", end = ' ')
    for i in range(N + 1):
        print(final_path[i], end = ' ')
    print(final_path)
    area = "Nibe-Gjøl Bredning Vildtreservat"
    data = extract_prot_area(area)
    bb_coords = data['bounds']
    protected_area = data['protected']
    lat1, lon1 = float(bb_coords[1]), float(bb_coords[0])
    lat2, lon2 = float(bb_coords[3]), float(bb_coords[2])
    grid_gdf = create_geographic_grid(lat1, lon1, lat2, lon2, sector_size_m=SECTOR_SIZE)
    #grid_gdf.geometry.intersects()

    grid_gdf['intersects'] = grid_gdf.geometry.intersects(Polygon(protected_area))
    # Count the number of True (intersecting) and False (non-intersecting) values
    intersecting_cells = grid_gdf['intersects'].sum()
    non_intersecting_cells = len(grid_gdf) - intersecting_cells
    all_cells = len(grid_gdf)

    print(f"Intersecting cells: {intersecting_cells}")
    print(f"Non-intersecting cells: {non_intersecting_cells}")
    print(f"all cells: {all_cells}")

    renderThread = threading.Thread(target=show_grid, args=(protected_area, grid_gdf, area))
    renderThread.start()

    #t2 = threading.Thread(target=print_cube, args=(10,))
    show_grid(protected_area, grid_gdf, area)

if __name__ == "__main__":
    
    main()