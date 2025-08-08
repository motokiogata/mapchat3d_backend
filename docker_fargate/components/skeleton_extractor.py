# skeleton_extractor.py
import cv2
import numpy as np
from skimage.morphology import medial_axis
from math import sqrt
from config import MIN_LINE_LENGTH, DEBUG_SKELETON

class SkeletonExtractor:
    def __init__(self):
        self.road_id_counter = 0
        self.intersection_id_counter = 0
    
    def extract_medial_axis(self, mask_path):
        """Extract skeleton from road mask"""
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {mask_path}")
        
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary = binary // 255
        skel = medial_axis(binary).astype(np.uint8)
        
        cv2.imwrite(DEBUG_SKELETON, skel * 255)
        print(f"✅ Skeleton extracted and saved to {DEBUG_SKELETON}")
        return skel

    def get_neighbors_8(self, y, x, skeleton):
        """Get 8-connected neighbors"""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] == 1:
                        neighbors.append((ny, nx))
        return neighbors

    def find_endpoints_and_junctions(self, skeleton):
        """Find endpoints and junctions with consistent labeling"""
        h, w = skeleton.shape
        endpoints = []
        junctions = []
        
        for y in range(h):
            for x in range(w):
                if skeleton[y, x] == 1:
                    neighbors = self.get_neighbors_8(y, x, skeleton)
                    degree = len(neighbors)
                    
                    if degree == 1:
                        endpoints.append((y, x))
                    elif degree > 2:
                        junctions.append((y, x))
        
        print(f"✅ Found {len(endpoints)} endpoints and {len(junctions)} junctions")
        return endpoints, junctions

    def trace_path_between_points(self, skeleton, start_y, start_x, visited, stop_points):
        """Trace path between points"""
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        
        current_y, current_x = start_y, start_x
        
        while True:
            neighbors = self.get_neighbors_8(current_y, current_x, skeleton)
            unvisited_neighbors = [(ny, nx) for ny, nx in neighbors if not visited[ny, nx]]
            
            if len(unvisited_neighbors) == 0:
                break
            elif len(unvisited_neighbors) == 1:
                next_y, next_x = unvisited_neighbors[0]
                visited[next_y, next_x] = True
                path.append((next_x, next_y))
                current_y, current_x = next_y, next_x
                
                if (next_y, next_x) in stop_points:
                    break
            else:
                break
        
        return path

    def trace_skeleton_paths(self, skeleton):
        """Trace skeleton paths with consistent road IDs"""
        visited = np.zeros_like(skeleton, dtype=bool)
        endpoints, junctions = self.find_endpoints_and_junctions(skeleton)
        
        stop_points = set(junctions + endpoints)
        roads = []
        
        # Trace from endpoints
        for start_y, start_x in endpoints:
            if not visited[start_y, start_x]:
                path = self.trace_path_between_points(skeleton, start_y, start_x, visited, stop_points)
                
                if len(path) >= MIN_LINE_LENGTH:
                    roads.append({
                        "id": self.road_id_counter,
                        "points": path,
                        "start_type": "endpoint",
                        "skeleton_endpoints": [(start_x, start_y)]
                    })
                    self.road_id_counter += 1
        
        # Trace from junctions
        for start_y, start_x in junctions:
            neighbors = self.get_neighbors_8(start_y, start_x, skeleton)
            for ny, nx in neighbors:
                if not visited[ny, nx]:
                    visited[start_y, start_x] = True
                    path = [(start_x, start_y)]
                    path.extend(self.trace_path_between_points(skeleton, ny, nx, visited, stop_points)[1:])
                    
                    if len(path) >= MIN_LINE_LENGTH:
                        roads.append({
                            "id": self.road_id_counter,
                            "points": path,
                            "start_type": "junction",
                            "skeleton_junctions": [(start_x, start_y)]
                        })
                        self.road_id_counter += 1
        
        return roads, endpoints, junctions

    def find_major_intersections(self, junctions, skeleton):
        """Find intersections with consistent intersection IDs"""
        if not junctions:
            return []
        
        major_intersections = []
        used = set()
        
        for i, (y1, x1) in enumerate(junctions):
            if (y1, x1) in used:
                continue
                
            cluster = [(y1, x1)]
            used.add((y1, x1))
            
            for j, (y2, x2) in enumerate(junctions):
                if i != j and (y2, x2) not in used:
                    dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                    if dist <= 15:  # Cluster nearby junctions
                        cluster.append((y2, x2))
                        used.add((y2, x2))
            
            if len(cluster) >= 1:
                center_y = int(np.mean([p[0] for p in cluster]))
                center_x = int(np.mean([p[1] for p in cluster]))
                
                roads_count = self.count_roads_at_intersection(center_y, center_x, skeleton)
                
                major_intersections.append({
                    'id': self.intersection_id_counter,
                    'center': (center_x, center_y),
                    'roads_count': roads_count,
                    'junction_points': [(x, y) for y, x in cluster],
                    'skeleton_junctions': cluster
                })
                self.intersection_id_counter += 1
        
        print(f"✅ Found {len(major_intersections)} major intersections with consistent IDs")
        return major_intersections

    def count_roads_at_intersection(self, center_y, center_x, skeleton, radius=10):
        """Count roads meeting at intersection"""
        directions = []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = center_y + dy, center_x + dx
                if (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1] and 
                    skeleton[y, x] == 1 and (dy != 0 or dx != 0)):
                    
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
        
        if not directions:
            return 0
        
        directions.sort()
        road_count = 1
        prev_angle = directions[0]
        
        for angle in directions[1:]:
            if abs(angle - prev_angle) > np.pi/4:
                road_count += 1
                prev_angle = angle
        
        if len(directions) > 1 and abs(directions[-1] - directions[0] + 2*np.pi) <= np.pi/4:
            road_count -= 1
        
        return max(road_count, 2)
4. geometry_utils.py - Geometry and Calculation Utils
# geometry_utils.py
import math
from math import sqrt, atan2, degrees
from config import CANVAS_SIZE, EDGE_TOLERANCE

class GeometryUtils:
    @staticmethod
    def calculate_road_direction(points):
        """Calculate road direction"""
        if len(points) < 2:
            return "unknown"
        
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        
        angle = math.degrees(math.atan2(-dy, dx))
        if angle < 0:
            angle += 360
        
        if 337.5 <= angle < 22.5 or angle == 0:
            cardinal = "East"
        elif 22.5 <= angle < 67.5:
            cardinal = "Northeast"
        elif 67.5 <= angle < 112.5:
            cardinal = "North"
        elif 112.5 <= angle < 157.5:
            cardinal = "Northwest"
        elif 157.5 <= angle < 202.5:
            cardinal = "West"
        elif 202.5 <= angle < 247.5:
            cardinal = "Southwest"
        elif 247.5 <= angle < 292.5:
            cardinal = "South"
        else:
            cardinal = "Southeast"
        
        simple = "North-South" if 45 <= angle < 135 or 225 <= angle < 315 else "East-West"
        
        return {"detailed": angle, "cardinal": cardinal, "simple": simple}

    @staticmethod
    def calculate_road_curvature(points):
        """Calculate road curvature"""
        if len(points) < 3:
            return "straight"
        
        angles = []
        for i in range(1, len(points)-1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.degrees(math.acos(cos_angle))
                angles.append(180 - angle)
        
        if not angles:
            return "straight"
        
        avg_angle = sum(angles) / len(angles)
        if avg_angle < 5:
            return "straight"
        elif avg_angle < 15:
            return "slight_curve"
        elif avg_angle < 30:
            return "moderate_curve"
        else:
            return "sharp_curve"

    @staticmethod
    def estimate_road_width_category(points):
        """Estimate road width category"""
        if len(points) < 50:
            return "narrow"
        elif len(points) < 150:
            return "medium"
        else:
            return "wide"

    @staticmethod
    def is_point_at_edge(point):
        """Analyze if point is at edge with detailed info"""
        x, y = point
        width, height = CANVAS_SIZE
        
        edge_info = {
            'is_edge': False,
            'edge_sides': [],
            'edge_distances': {},
            'edge_coordinates': {},
            'edge_id': None
        }
        
        distances = {
            'west': x,
            'east': width - x,
            'north': y,
            'south': height - y
        }
        
        for side, distance in distances.items():
            if distance <= EDGE_TOLERANCE:
                edge_info['is_edge'] = True
                edge_info['edge_sides'].append(side)
                edge_info['edge_distances'][side] = distance
                
                if side == 'west':
                    edge_info['edge_coordinates'][side] = {'x': 0, 'y': y}
                elif side == 'east':
                    edge_info['edge_coordinates'][side] = {'x': width, 'y': y}
                elif side == 'north':
                    edge_info['edge_coordinates'][side] = {'x': x, 'y': 0}
                elif side == 'south':
                    edge_info['edge_coordinates'][side] = {'x': x, 'y': height}
        
        return edge_info

    @staticmethod
    def calculate_turn_type(from_road, to_road, intersection_center):
        """Calculate turn type between roads"""
        if not from_road or not to_road:
            return "unknown"
        
        # Get vectors from intersection to road endpoints
        from_points = from_road['points']
        to_points = to_road['points']
        
        # Find closest points to intersection
        from_point = min(from_points, key=lambda p: sqrt((p[0] - intersection_center[0])**2 + (p[1] - intersection_center[1])**2))
        to_point = min(to_points, key=lambda p: sqrt((p[0] - intersection_center[0])**2 + (p[1] - intersection_center[1])**2))
        
        # Calculate angles
        dx1 = from_point[0] - intersection_center[0]
        dy1 = from_point[1] - intersection_center[1]
        dx2 = to_point[0] - intersection_center[0]
        dy2 = to_point[1] - intersection_center[1]
        
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        
        angle_diff = angle2 - angle1
        angle_diff = math.degrees(angle_diff)
        
        # Normalize to 0-360
        if angle_diff < 0:
            angle_diff += 360
        
        # Classify turn
        if 315 <= angle_diff or angle_diff <= 45:
            return "straight"
        elif 45 < angle_diff <= 135:
            return "right"
        elif 135 < angle_diff <= 225:
            return "u_turn"
        else:
            return "left"

    @staticmethod
    def get_clock_direction(center, point):
        """Get clock direction from center to point"""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = atan2(-dy, dx)
        
        angle_deg = degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        clock_hour = int((angle_deg + 15) / 30) % 12
        return clock_hour

    @staticmethod
    def determine_intersection_type(roads_count):
        """Determine intersection type"""
        if roads_count == 2:
            return "dead_end"
        elif roads_count == 3:
            return "T-intersection"
        elif roads_count == 4:
            return "4-way_cross"
        else:
            return "complex_intersection"