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