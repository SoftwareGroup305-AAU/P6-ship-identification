import math
import numpy as np

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    return vector / magnitude

def yaw_matrix(yaw):
    return np.array([
        [math.cos(yaw), -math.sin(yaw), 0], 
        [math.sin(yaw), math.cos(yaw), 0], 
        [0, 0, 1]
    ], dtype=float)

def pitch_matrix(pitch):
    return np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=float)

def roll_matrix(roll):
    return np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ], dtype=float)

def rotation_matrix(yaw, pitch, roll):
    return yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll)

def ray_top_right(fov_h, fov_v):
    vector = np.array([math.tan(fov_h / 2), math.tan(fov_v / 2), -1])
    return normalize(vector)

def ray_top_left(fov_h, fov_v):
    vector = np.array([-math.tan(fov_h / 2), math.tan(fov_v / 2), -1])
    return normalize(vector)

def ray_bottom_right(fov_h, fov_v):
    vector = np.array([math.tan(fov_h / 2), -math.tan(fov_v / 2), -1])
    return normalize(vector)

def ray_bottom_left(fov_h, fov_v):
    vector = np.array([-math.tan(fov_h / 2), -math.tan(fov_v / 2), -1])
    return normalize(vector)

def get_ray_intersections(rays, origin):
    intersections = []
    for ray in rays:
        # Extract the origin (x, y, z) and direction (dx, dy, dz) components
        o_x, o_y, o_z = origin
        d_x, d_y, d_z = ray
        
        # Skip if ray does not intersect the the ground (d_z >= 0)
        if d_z >= 0:
            continue
        
        # Calculate the t value (the intersection factor)
        t = -o_z / d_z  # z = 0 (ground plane)

        # Calculate the intersection point: p = o + t * d
        p_x = o_x + t * d_x
        p_y = o_y + t * d_y
        p_z = 0  # We know the intersection happens at z = 0

        # Store the intersection point
        intersections.append(np.array([p_x, p_y, p_z]))
    
    return intersections

def get_bounding_area(x, y, fov_h, fov_v, altitude, roll, pitch, yaw):
    rotation = rotation_matrix(yaw, pitch, roll)

    ray_1 = rotation @ ray_top_right(fov_h, fov_v) 
    ray_2 = rotation @ ray_top_left(fov_h, fov_v) 
    ray_3 = rotation @ ray_bottom_right(fov_h, fov_v)
    ray_4 = rotation @ ray_bottom_left(fov_h, fov_v) 

    origin = np.array([x, y, altitude])

    intersections = get_ray_intersections([ray_1, ray_2, ray_3, ray_4], origin)
    return intersections

def test_get_bounding_area():
    x, y = 0, 0  # Drone positioned at the origin
    altitude = 50  # Drone is 150 meters above the ground
    fov_h = math.radians(90)  # 90-degree horizontal FOV
    fov_v = math.radians(60)  # 60-degree vertical FOV
    roll, pitch, yaw = 0, math.radians(90), 0 # 30° pitch, 15° yaw

    bounding_area = get_bounding_area(x, y, fov_h, fov_v, altitude, roll, pitch, yaw)

    print("Bounding Area Points (Intersections with Ground):")
    for i, point in enumerate(bounding_area):
        print(f"Point {i+1}: {point}")

test_get_bounding_area()