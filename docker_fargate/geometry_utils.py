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