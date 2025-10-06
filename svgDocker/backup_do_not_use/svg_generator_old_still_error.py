#svg_generator.py
import json
import boto3
import os
import logging
import svgwrite
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import sys
import math
from typing import Optional, Union
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -------------------------
# 🔧 Small helpers (KEEP)
# -------------------------
def _get_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def _load_json_from_s3(bucket: str, key: str) -> dict:
    s3 = boto3.client("s3")
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

@dataclass
class VehiclePath:
    vehicle_id: str
    start_lane_id: str
    end_lane_id: str
    maneuver: str
    route_points: List[Tuple[float, float]]
    collision_point: Tuple[float, float]
    timeline: List[Dict]

@dataclass
class CollisionScenario:
    user_vehicle: VehiclePath
    other_vehicle: VehiclePath
    collision_point: Tuple[float, float]
    collision_type: str
    base_map_image: str

# ❌ REMOVE: EnhancedTwoVehicleAnimationGenerator (DEAD CODE)
# This entire class is broken - DELETE IT

# ✅ KEEP: SVGAnimationGenerator (WORKING - unchanged)
class SVGAnimationGenerator:
    def __init__(self, s3_bucket: str, connection_id: str):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.connection_id = connection_id
        self.lane_tree_data = None
        self.base_map_bounds = None
        self.svg_width = 800
        self.svg_height = 600
        
    # ... ALL YOUR EXISTING METHODS UNCHANGED ...
    def generate_accident_animation(self, route_json: Dict) -> str:
        # ... keep exactly as is ...
        pass
    
    def load_lane_tree_data(self):
        # ... keep exactly as is ...
        pass
    
    # ... keep all other methods exactly as they are ...

# 🔧 FIX: EnhancedSVGAnimationGenerator (ADD MISSING METHODS)
class EnhancedSVGAnimationGenerator:

    def __init__(self, s3_bucket: str, connection_id: str):
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client("bedrock-runtime", 
                                   region_name=os.environ.get("AWS_REGION", "us-east-1"))
        self.bucket = s3_bucket
        self.connection_id = connection_id
        self.svg_width = 1200
        self.svg_height = 800
        
        # 🆕 ADD: Analytics support
        self.analytics_metadata = None
        self.lane_tree_data = None

    def generate_guided_vehicle_paths(self, scenario_analysis: dict, full_data: dict) -> dict:
        """Generate vehicle paths using LLM guidance and full waypoint data"""
        logger.info("🛣️ Generating guided vehicle paths")
        
        try:
            vehicle_paths = {}
            
            # Extract vehicle 1 (user vehicle)
            vehicle_1_data = scenario_analysis.get('vehicle_1', {})
            if vehicle_1_data:
                lane_id = vehicle_1_data.get('best_lane_id', 'fallback_1')
                waypoints = self.extract_waypoints_for_lane(lane_id, full_data)
                
                if not waypoints:
                    waypoints = self.create_fallback_waypoints('user')
                
                vehicle_paths['user_vehicle'] = {
                    'waypoints': self.convert_waypoints_to_svg(waypoints),
                    'raw_waypoints': waypoints,
                    'color': 'blue',
                    'label': 'Your Vehicle',
                    'lane_id': lane_id,
                    'maneuver': vehicle_1_data.get('maneuver', 'straight'),
                    'collision_timing': vehicle_1_data.get('collision_timing', '60%')
                }
            
            # Extract vehicle 2 (other vehicle)
            vehicle_2_data = scenario_analysis.get('vehicle_2', {})
            if vehicle_2_data:
                lane_id = vehicle_2_data.get('best_lane_id', 'fallback_2')
                waypoints = self.extract_waypoints_for_lane(lane_id, full_data)
                
                if not waypoints:
                    waypoints = self.create_fallback_waypoints('other')
                
                vehicle_paths['other_vehicle'] = {
                    'waypoints': self.convert_waypoints_to_svg(waypoints),
                    'raw_waypoints': waypoints,
                    'color': 'red',
                    'label': 'Other Vehicle',
                    'lane_id': lane_id,
                    'maneuver': vehicle_2_data.get('maneuver', 'straight'),
                    'collision_timing': vehicle_2_data.get('collision_timing', '65%')
                }
            
            logger.info(f"✅ Generated paths for {len(vehicle_paths)} vehicles")
            return vehicle_paths
            
        except Exception as e:
            logger.error(f"❗ Failed to generate guided vehicle paths: {e}")
            return self.create_fallback_vehicle_paths()

    def create_fallback_waypoints(self, vehicle_type: str) -> list:
        """Create fallback waypoints when lane extraction fails"""
        if vehicle_type == 'user':
            return [
                {'x': 100, 'y': self.svg_height // 2},
                {'x': self.svg_width // 2 - 50, 'y': self.svg_height // 2},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2}
            ]
        else:  # other vehicle
            return [
                {'x': self.svg_width // 2, 'y': 100},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2 - 50},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2}
            ]

    def create_fallback_vehicle_paths(self) -> dict:
        """Create fallback paths for both vehicles when everything fails"""
        return {
            'user_vehicle': {
                'waypoints': [
                    {'x': 100, 'y': self.svg_height // 2},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ],
                'color': 'blue',
                'label': 'Your Vehicle (fallback)',
                'maneuver': 'straight'
            },
            'other_vehicle': {
                'waypoints': [
                    {'x': self.svg_width // 2, 'y': 100},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ],
                'color': 'red', 
                'label': 'Other Vehicle (fallback)',
                'maneuver': 'straight'
            }
        }

    def create_dual_vehicle_svg_animation(self, user_path: dict, other_path: dict, collision_data: dict) -> str:
        """Create SVG with two vehicles moving toward collision"""
        logger.info("🎬 Creating dual-vehicle SVG animation")
        
        dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(self.svg_width, self.svg_height), fill='#1a1a1a'))
        
        # Add intersection background
        self.add_intersection_background(dwg)
        
        # Add vehicle paths (visual guides)
        self.add_vehicle_path_visualization(dwg, user_path, other_path)
        
        # Add both vehicle animations
        self.add_vehicle_animation(dwg, user_path, collision_data, 'user')
        self.add_vehicle_animation(dwg, other_path, collision_data, 'other')
        
        # Add collision effect
        self.add_collision_effect(dwg, collision_data)
        
        # Add labels and timeline
        self.add_dual_vehicle_labels(dwg, user_path, other_path)
        self.add_timeline_display(dwg)
        
        return dwg.tostring()

    def add_vehicle_path_visualization(self, dwg, user_path: dict, other_path: dict):
        """Add visual representation of both vehicle paths"""
        # User vehicle path (blue, dashed)
        user_waypoints = user_path.get('waypoints', [])
        if len(user_waypoints) > 1:
            path_data = self.waypoints_to_svg_path(user_waypoints)
            dwg.add(dwg.path(
                d=path_data,
                stroke='lightblue',
                stroke_width=2,
                stroke_dasharray="5,5",
                fill='none',
                opacity=0.6
            ))
        
        # Other vehicle path (red, dashed)
        other_waypoints = other_path.get('waypoints', [])
        if len(other_waypoints) > 1:
            path_data = self.waypoints_to_svg_path(other_waypoints)
            dwg.add(dwg.path(
                d=path_data,
                stroke='lightcoral',
                stroke_width=2,
                stroke_dasharray="5,5",
                fill='none',
                opacity=0.6
            ))

    def add_vehicle_animation(self, dwg, vehicle_path: dict, collision_data: dict, vehicle_type: str):
        """Add animated vehicle to SVG"""
        waypoints = vehicle_path.get('waypoints', [])
        if not waypoints:
            return
        
        color = vehicle_path.get('color', 'gray')
        vehicle_size = 12 if vehicle_type == 'user' else 10
        
        # Create vehicle shape (circle for simplicity)
        vehicle = dwg.circle(
            center=(waypoints[0]['x'] if isinstance(waypoints[0], dict) else waypoints[0][0],
                    waypoints[0]['y'] if isinstance(waypoints[0], dict) else waypoints[0][1]),
            r=vehicle_size,
            fill=color,
            stroke='white',
            stroke_width=2
        )
        
        # Create animation path
        if len(waypoints) > 1:
            animation_path = self.waypoints_to_svg_path(waypoints)
            
            # Calculate animation duration based on collision timing
            collision_progress = collision_data.get(f'{vehicle_type}_collision_progress', 0.6)
            total_duration = 8.0  # Total animation time in seconds
            
            animate_motion = dwg.animateMotion(
                path=animation_path,
                dur=f"{total_duration}s",
                repeatCount="1",
                fill="freeze"
            )
            
            vehicle.add(animate_motion)
        
        dwg.add(vehicle)
        
        # Add vehicle direction indicator
        if len(waypoints) > 1:
            start_point = waypoints[0]
            end_point = waypoints[1]
            
            if isinstance(start_point, dict):
                start_x, start_y = start_point['x'], start_point['y']
            else:
                start_x, start_y = start_point[0], start_point[1]
                
            if isinstance(end_point, dict):
                end_x, end_y = end_point['x'], end_point['y']
            else:
                end_x, end_y = end_point[0], end_point[1]
            
            # Calculate direction arrow
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                dx /= length
                dy /= length
                
                arrow_end_x = start_x + dx * 30
                arrow_end_y = start_y + dy * 30
                
                dwg.add(dwg.line(
                    start=(start_x, start_y),
                    end=(arrow_end_x, arrow_end_y),
                    stroke=color,
                    stroke_width=3,
                    opacity=0.7
                ))

    def waypoints_to_svg_path(self, waypoints: list) -> str:
        """Convert waypoints to SVG path data"""
        if not waypoints:
            return ""
        
        # Handle first point
        first_point = waypoints[0]
        if isinstance(first_point, dict):
            path_data = f"M{first_point['x']},{first_point['y']}"
        else:
            path_data = f"M{first_point[0]},{first_point[1]}"
        
        # Add remaining points
        for point in waypoints[1:]:
            if isinstance(point, dict):
                path_data += f" L{point['x']},{point['y']}"
            else:
                path_data += f" L{point[0]},{point[1]}"
        
        return path_data

    def add_collision_effect(self, dwg, collision_data: dict):
        """Add collision effect animation"""
        collision_point = collision_data.get('collision_point', [self.svg_width//2, self.svg_height//2])
        collision_timing = collision_data.get('collision_timing', 5.0)
        
        # Explosion effect
        explosion = dwg.circle(
            center=(collision_point[0], collision_point[1]),
            r=5,
            fill='orange',
            opacity=0
        )
        
        # Animate explosion
        explosion.add(dwg.animate(
            attributeName='r',
            values='5;50;30',
            dur='2s',
            begin=f'{collision_timing}s',
            repeatCount='1'
        ))
        
        explosion.add(dwg.animate(
            attributeName='opacity',
            values='0;1;0.5;0',
            dur='2s',
            begin=f'{collision_timing}s',
            repeatCount='1'
        ))
        
        dwg.add(explosion)
        
        # Add collision warning text
        warning_text = dwg.text(
            "COLLISION!",
            insert=(collision_point[0], collision_point[1] - 60),
            text_anchor="middle",
            fill='red',
            font_size=20,
            font_weight='bold',
            opacity=0
        )
        
        warning_text.add(dwg.animate(
            attributeName='opacity',
            values='0;1;1;0',
            dur='3s',
            begin=f'{collision_timing}s',
            repeatCount='1'
        ))
        
        dwg.add(warning_text)

    def add_dual_vehicle_labels(self, dwg, user_path: dict, other_path: dict):
        """Add labels for both vehicles"""
        # User vehicle label
        user_waypoints = user_path.get('waypoints', [])
        if user_waypoints:
            start_point = user_waypoints[0]
            if isinstance(start_point, dict):
                x, y = start_point['x'], start_point['y']
            else:
                x, y = start_point[0], start_point[1]
            
            dwg.add(dwg.text(
                user_path.get('label', 'Your Vehicle'),
                insert=(x - 40, y - 30),
                fill='lightblue',
                font_size=14,
                font_weight='bold'
            ))
        
        # Other vehicle label
        other_waypoints = other_path.get('waypoints', [])
        if other_waypoints:
            start_point = other_waypoints[0]
            if isinstance(start_point, dict):
                x, y = start_point['x'], start_point['y']
            else:
                x, y = start_point[0], start_point[1]
            
            dwg.add(dwg.text(
                other_path.get('label', 'Other Vehicle'),
                insert=(x - 40, y - 30),
                fill='lightcoral',
                font_size=14,
                font_weight='bold'
            ))

    def add_timeline_display(self, dwg):
        """Add timeline display at bottom"""
        # Timeline background
        timeline_bg = dwg.rect(
            insert=(50, self.svg_height - 100),
            size=(self.svg_width - 100, 80),
            fill='black',
            opacity=0.8,
            rx=10
        )
        dwg.add(timeline_bg)
        
        # Timeline text
        dwg.add(dwg.text(
            "Two-Vehicle Collision Animation",
            insert=(self.svg_width//2, self.svg_height - 70),
            text_anchor="middle",
            fill='white',
            font_size=18,
            font_weight='bold'
        ))
        
        dwg.add(dwg.text(
            "Blue: Your Vehicle | Red: Other Vehicle | Orange: Collision Point",
            insert=(self.svg_width//2, self.svg_height - 45),
            text_anchor="middle",
            fill='lightgray',
            font_size=12
        ))


    def calculate_collision_timing(self, user_path: dict, other_path: dict) -> dict:
        """Calculate when and where vehicles collide"""
        logger.info("💥 Calculating collision timing and point")
        
        user_waypoints = user_path.get('waypoints', [])
        other_waypoints = other_path.get('waypoints', [])
        
        if not user_waypoints or not other_waypoints:
            return self.create_default_collision()
        
        # Find intersection point of paths
        collision_point = self.find_path_intersection(user_waypoints, other_waypoints)
        
        # Calculate timing - when each vehicle reaches collision point
        user_collision_time = self.calculate_time_to_point(user_waypoints, collision_point)
        other_collision_time = self.calculate_time_to_point(other_waypoints, collision_point)
        
        # Synchronize timing for collision
        collision_timing = max(user_collision_time, other_collision_time)
        
        return {
            'collision_point': collision_point,
            'collision_timing': collision_timing,
            'user_collision_progress': user_collision_time / len(user_waypoints) if user_waypoints else 0.5,
            'other_collision_progress': other_collision_time / len(other_waypoints) if other_waypoints else 0.5,
            'collision_type': 'intersection_collision'
        }

    def find_path_intersection(self, path1: list, path2: list) -> list:
        """Find where two paths intersect (approximate)"""
        min_distance = float('inf')
        intersection_point = [self.svg_width // 2, self.svg_height // 2]  # Default center
        
        # Check all combinations of points
        for p1 in path1:
            for p2 in path2:
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    # Average of closest points
                    intersection_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        
        logger.info(f"🎯 Collision point: {intersection_point}, closest distance: {min_distance:.1f}")
        return intersection_point

    def calculate_time_to_point(self, waypoints: list, target_point: list) -> int:
        """Calculate how many steps it takes to reach target point"""
        if not waypoints or not target_point:
            return len(waypoints) // 2 if waypoints else 10
        
        min_distance = float('inf')
        closest_index = len(waypoints) // 2  # Default middle
        
        for i, point in enumerate(waypoints):
            if isinstance(point, dict):
                point_coords = [point.get('x', 0), point.get('y', 0)]
            else:
                point_coords = point
                
            distance = math.sqrt((point_coords[0] - target_point[0])**2 + (point_coords[1] - target_point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index

    def create_default_collision(self) -> dict:
        """Create default collision data when calculation fails"""
        return {
            'collision_point': [self.svg_width // 2, self.svg_height // 2],
            'collision_timing': 5.0,
            'user_collision_progress': 0.6,
            'other_collision_progress': 0.6,
            'collision_type': 'intersection_collision'
        }

    def extract_vehicle_path_from_metadata(self, accident_scenario: dict, vehicle_type: str) -> dict:
        """Extract clean waypoint sequence from accident scenario metadata"""
        logger.info(f"🚗 Extracting {vehicle_type} vehicle path from metadata")
        
        try:
            # Get vehicle data from accident scenario
            vehicle_data = accident_scenario.get(f'{vehicle_type}_vehicle', {})
            if not vehicle_data:
                logger.warning(f"⚠️ No {vehicle_type} vehicle data found")
                return self.create_fallback_vehicle_path(vehicle_type)
            
            # Extract path data
            path_data = vehicle_data.get('path', {})
            
            # Option 1: Use route_points if available
            route_points = path_data.get('route_points', [])
            if route_points:
                waypoints = self.process_route_points(route_points)
                logger.info(f"✅ Found {len(waypoints)} route points for {vehicle_type}")
            else:
                # Option 2: Use lane_id to get waypoints
                lane_id = path_data.get('lane_id') or vehicle_data.get('lane_id')
                if lane_id:
                    waypoints = self.extract_waypoints_for_lane(str(lane_id), {})
                    logger.info(f"✅ Extracted {len(waypoints)} waypoints from lane {lane_id} for {vehicle_type}")
                else:
                    # Option 3: Create path from origin/destination
                    waypoints = self.create_path_from_directions(path_data, vehicle_type)
                    logger.info(f"✅ Created {len(waypoints)} waypoints from directions for {vehicle_type}")
            
            # Apply smooth connection logic to prevent jumping
            smooth_waypoints = self.smooth_waypoint_sequence(waypoints)
            
            # Convert to SVG coordinates
            svg_waypoints = self.convert_waypoints_to_svg(smooth_waypoints)
            
            return {
                'waypoints': svg_waypoints,
                'raw_waypoints': waypoints,
                'vehicle_type': vehicle_type,
                'color': 'blue' if vehicle_type == 'user' else 'red',
                'label': 'Your Vehicle' if vehicle_type == 'user' else 'Other Vehicle',
                'metadata': vehicle_data,
                'path_quality': 'smooth_connected'
            }
            
        except Exception as e:
            logger.error(f"❗ Error extracting {vehicle_type} vehicle path: {e}")
            return self.create_fallback_vehicle_path(vehicle_type)



    def extract_vehicle_path_from_metadata(self, accident_scenario: dict, vehicle_type: str) -> dict:
        """Extract vehicle path using algorithmic lane determination"""
        logger.info(f"🚗 Extracting {vehicle_type} vehicle path with lane algorithm")
        
        try:
            vehicle_data = accident_scenario.get(f'{vehicle_type}_vehicle', {})
            if not vehicle_data:
                return self.create_fallback_vehicle_path(vehicle_type)
            
            # Get road ID from metadata
            path_data = vehicle_data.get('path', {})
            road_id = path_data.get('origin', {}).get('road_id', 'unknown')
            
            if road_id == 'unknown' or road_id == -1:
                logger.warning(f"⚠️ Unknown road_id for {vehicle_type}, using fallback")
                return self.create_fallback_vehicle_path(vehicle_type)
            
            # 🎯 ALGORITHM: Determine correct lane based on direction
            correct_lane_id = self.determine_correct_lane(road_id, vehicle_type, vehicle_data)
            
            # Extract waypoints for the determined lane
            waypoints = self.extract_waypoints_for_lane(correct_lane_id, {})
            
            if not waypoints:
                logger.warning(f"⚠️ No waypoints found for lane {correct_lane_id}")
                waypoints = self.create_fallback_waypoints(vehicle_type)
            
            # Convert to SVG coordinates
            svg_waypoints = self.convert_waypoints_to_svg(waypoints)
            
            return {
                'waypoints': svg_waypoints,
                'vehicle_type': vehicle_type,
                'color': 'blue' if vehicle_type == 'user' else 'red',
                'label': 'Your Vehicle' if vehicle_type == 'user' else 'Other Vehicle',
                'determined_lane': correct_lane_id,
                'path_quality': 'algorithmic_lane_selection'
            }
            
        except Exception as e:
            logger.error(f"❗ Error in algorithmic lane extraction for {vehicle_type}: {e}")
            return self.create_fallback_vehicle_path(vehicle_type)


    def process_route_points(self, route_points: list) -> list:
        """Process route points from metadata into clean waypoints"""
        waypoints = []
        
        for point in route_points:
            if isinstance(point, dict):
                if 'x' in point and 'y' in point:
                    waypoints.append([float(point['x']), float(point['y'])])
                elif 'coordinates' in point:
                    coords = point['coordinates']
                    if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                        waypoints.append([float(coords['x']), float(coords['y'])])
                elif 'lat' in point and 'lng' in point:
                    # Convert lat/lng to x/y (simple conversion)
                    x = float(point['lng']) * 100000  # Simple scaling
                    y = float(point['lat']) * 100000
                    waypoints.append([x, y])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                waypoints.append([float(point[0]), float(point[1])])
        
        return waypoints

    def create_path_from_directions(self, path_data: dict, vehicle_type: str) -> list:
        """Create waypoint path from origin/destination directions"""
        origin = path_data.get('origin', {})
        destination = path_data.get('intended_destination', {})
        
        # Map directions to canvas positions
        direction_positions = {
            'north': (self.svg_width // 2, 50),
            'south': (self.svg_width // 2, self.svg_height - 50),
            'east': (self.svg_width - 50, self.svg_height // 2),
            'west': (50, self.svg_height // 2),
            'northeast': (self.svg_width - 50, 50),
            'northwest': (50, 50),
            'southeast': (self.svg_width - 50, self.svg_height - 50),
            'southwest': (50, self.svg_height - 50)
        }
        
        # Get origin position
        origin_dir = origin.get('direction', 'west').lower()
        start_pos = direction_positions.get(origin_dir, (100, self.svg_height // 2))
        
        # Get destination based on maneuver
        maneuver = destination.get('maneuver', 'straight').lower()
        center = (self.svg_width // 2, self.svg_height // 2)
        
        if maneuver == 'straight':
            if origin_dir in ['west', 'east']:
                end_pos = (self.svg_width - start_pos[0], start_pos[1])
            else:
                end_pos = (start_pos[0], self.svg_height - start_pos[1])
        elif maneuver in ['left', 'turn_left']:
            end_pos = self.calculate_left_turn_destination(start_pos, origin_dir)
        elif maneuver in ['right', 'turn_right']:
            end_pos = self.calculate_right_turn_destination(start_pos, origin_dir)
        else:
            end_pos = center
        
        # Create path with multiple waypoints
        return self.create_smooth_connection(list(start_pos), list(end_pos))

    def calculate_left_turn_destination(self, start_pos: tuple, direction: str) -> tuple:
        """Calculate destination for left turn"""
        center_x, center_y = self.svg_width // 2, self.svg_height // 2
        
        if direction == 'west':
            return (center_x, self.svg_height - 50)  # Turn south
        elif direction == 'east':
            return (center_x, 50)  # Turn north
        elif direction == 'north':
            return (50, center_y)  # Turn west
        elif direction == 'south':
            return (self.svg_width - 50, center_y)  # Turn east
        else:
            return (center_x, center_y)

    def calculate_right_turn_destination(self, start_pos: tuple, direction: str) -> tuple:
        """Calculate destination for right turn"""
        center_x, center_y = self.svg_width // 2, self.svg_height // 2
        
        if direction == 'west':
            return (center_x, 50)  # Turn north
        elif direction == 'east':
            return (center_x, self.svg_height - 50)  # Turn south
        elif direction == 'north':
            return (self.svg_width - 50, center_y)  # Turn east
        elif direction == 'south':
            return (50, center_y)  # Turn west
        else:
            return (center_x, center_y)

    def create_fallback_vehicle_path(self, vehicle_type: str) -> dict:
        """Create fallback path when extraction fails"""
        if vehicle_type == 'user':
            waypoints = [[100, self.svg_height // 2], [self.svg_width // 2, self.svg_height // 2]]
            color = 'blue'
            label = 'Your Vehicle (fallback)'
        else:
            waypoints = [[self.svg_width // 2, 100], [self.svg_width // 2, self.svg_height // 2]]
            color = 'red'
            label = 'Other Vehicle (fallback)'
        
        return {
            'waypoints': waypoints,
            'vehicle_type': vehicle_type,
            'color': color,
            'label': label,
            'path_quality': 'fallback'
        }

    def create_smooth_connection(self, point1: list, point2: list) -> list:
        """Create smooth connection curve between two points (ported from local)"""
        if not point1 or not point2 or len(point1) < 2 or len(point2) < 2:
            return []
        
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
        if distance < 5:
            return [point1, point2]
        
        # Create smooth curve with 3-5 intermediate points
        num_points = min(5, max(3, int(distance / 20)))
        connection_curve = [point1]
        
        for i in range(1, num_points):
            t = i / num_points
            # Simple linear interpolation (can be enhanced with curves later)
            x = point1[0] + t * (point2[0] - point1[0])
            y = point1[1] + t * (point2[1] - point1[1])
            connection_curve.append([x, y])
        
        connection_curve.append(point2)
        return connection_curve

    def smooth_waypoint_sequence(self, waypoints: list) -> list:
        """Prevent jumping by ensuring continuous waypoint connections"""
        if not waypoints or len(waypoints) < 2:
            return waypoints
        
        smooth_sequence = [waypoints[0]]
        CONNECTION_THRESHOLD = 50  # Max distance before adding smooth connection
        
        for i in range(1, len(waypoints)):
            prev_point = smooth_sequence[-1]
            curr_point = waypoints[i]
            
            # Check if points are too far apart (potential jump)
            distance = math.sqrt((prev_point[0] - curr_point[0])**2 + (prev_point[1] - curr_point[1])**2)
            
            if distance > CONNECTION_THRESHOLD:
                # Add smooth connection points
                connection_points = self.create_smooth_connection(prev_point, curr_point)
                smooth_sequence.extend(connection_points[1:])  # Skip first point (duplicate)
            else:
                smooth_sequence.append(curr_point)
        
        return smooth_sequence
        
    def generate_intelligent_animation(self) -> str:
        """Main method - choose mode based on environment"""
        try:
            mode = os.environ.get("MODE", "ENHANCED_LLM_GUIDED")
            
            if mode == "ANALYTICS_GUIDED":
                logger.info("🎯 Using ANALYTICS-GUIDED mode")
                return self.generate_analytics_guided_animation()
            else:
                logger.info("🤖 Using LLM-GUIDED mode")
                return self.generate_llm_guided_animation()
                
        except Exception as e:
            logger.error(f"❗ Enhanced animation generation failed: {e}")
            return self.create_fallback_animation()


    def debug_accident_scenario(self, llm_data: dict):
        """Debug method to inspect accident scenario structure"""
        logger.info("🔍 DEBUG: Accident scenario structure:")
        
        accident_scenario = llm_data.get('accident_scenario', {})
        
        for key, value in accident_scenario.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        logger.info(f"    {subkey}: {list(subvalue.keys())}")
                    else:
                        logger.info(f"    {subkey}: {type(subvalue)} - {str(subvalue)[:100]}")
            else:
                logger.info(f"  {key}: {type(value)} - {str(value)[:100]}")


    def generate_analytics_guided_animation(self) -> str:
        """Enhanced: Generate dual-vehicle animation using precise analytics data"""
        logger.info("🎯 Starting enhanced dual-vehicle analytics-guided animation")
        
        try:
            # Load analytics metadata
            llm_data = self.load_llm_ready_data()
            self.debug_accident_scenario(llm_data)  # DEBUG: Inspect data structure
            self.analytics_metadata = llm_data
            
            # Extract accident scenario
            accident_scenario = llm_data.get('accident_scenario', {})
            if not accident_scenario:
                logger.warning("⚠️ No accident scenario found, using fallback")
                return self.create_fallback_animation()
            
            # Extract both vehicle paths with smooth connections
            logger.info("🚗 Extracting user vehicle path...")
            user_path = self.extract_vehicle_path_from_metadata(accident_scenario, 'user')
            
            logger.info("🚙 Extracting other vehicle path...")
            other_path = self.extract_vehicle_path_from_metadata(accident_scenario, 'other')
            
            # Calculate collision timing and point
            logger.info("💥 Calculating collision dynamics...")
            collision_data = self.calculate_collision_timing(user_path, other_path)
            
            # Create dual-vehicle SVG animation
            logger.info("🎬 Creating dual-vehicle SVG animation...")
            svg_content = self.create_dual_vehicle_svg_animation(user_path, other_path, collision_data)
            
            # Save to S3
            svg_key = f"animations/{self.connection_id}/dual_vehicle_analytics_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=svg_content,
                ContentType='image/svg+xml'
            )
            
            logger.info(f"✅ Dual-vehicle analytics SVG saved: {svg_key}")
            logger.info(f"📊 User path: {len(user_path.get('waypoints', []))} waypoints")
            logger.info(f"📊 Other path: {len(other_path.get('waypoints', []))} waypoints")
            logger.info(f"💥 Collision at: {collision_data.get('collision_point')}")
            
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ Enhanced dual-vehicle animation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.create_fallback_animation()


    # 🔄 RENAME: Your existing method
    def generate_llm_guided_animation(self) -> str:
        """Your existing LLM-guided method (renamed)"""
        # 1. Load LLM-ready data (metadata only)
        llm_data = self.load_llm_ready_data()
        
        # 2. Use LLM to understand the accident scenario
        scenario_analysis = self.analyze_scenario_with_llm(llm_data)
        
        # 3. Load full waypoint data for actual animation
        full_data = self.load_full_route_data()
        
        # 4. Generate vehicle paths using LLM guidance + full waypoints
        vehicle_paths = self.generate_guided_vehicle_paths(scenario_analysis, full_data)
        
        # 5. Create multi-vehicle animated SVG
        svg_content = self.create_multi_vehicle_animation(vehicle_paths, scenario_analysis)
        
        # 6. Save to S3
        svg_key = f"animations/{self.connection_id}/intelligent_animation.svg"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=svg_key,
            Body=svg_content,
            ContentType='image/svg+xml'
        )
        
        logger.info(f"✅ Generated intelligent SVG animation: {svg_key}")
        return svg_key

    # ✅ KEEP: Your existing methods (unchanged)
    def load_llm_ready_data(self) -> dict:
        """Load the lightweight metadata-only data"""
        llm_key = f"animation-data/{self.connection_id}/llm_ready_data.json"
        response = self.s3.get_object(Bucket=self.bucket, Key=llm_key)
        return json.loads(response['Body'].read().decode('utf-8'))

    def analyze_scenario_with_llm(self, llm_data: dict) -> dict:
        # ... keep exactly as is ...
        pass

    def create_multi_vehicle_animation(self, vehicle_paths: dict, scenario_analysis: dict) -> str:
        """Create SVG animation with multiple vehicles"""
        logger.info("🎬 Creating multi-vehicle SVG animation")
        
        dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(self.svg_width, self.svg_height), fill='#1a1a1a'))
        
        # Add intersection
        self.add_intersection_background(dwg)
        
        # Add each vehicle animation
        for vehicle_id, path_data in vehicle_paths.items():
            self.add_enhanced_vehicle_animation(dwg, path_data, scenario_analysis)
        
        # Add collision animation
        collision_data = scenario_analysis.get('collision', {})
        self.add_collision_animation(dwg, collision_data)
        
        # Add timeline
        animation_guidance = scenario_analysis.get('animation_guidance', {})
        self.add_enhanced_timeline(dwg, animation_guidance)
        
        return dwg.tostring()

    def add_enhanced_vehicle_animation(self, dwg, path_data: dict, scenario_analysis: dict):
        """Add enhanced vehicle animation to SVG"""
        waypoints = path_data.get('waypoints', [])
        if not waypoints:
            return
        
        color = path_data.get('color', 'gray')
        label = path_data.get('label', 'Vehicle')
        
        # Create vehicle shape
        start_point = waypoints[0]
        vehicle = dwg.circle(
            center=(start_point['x'], start_point['y']),
            r=12,
            fill=color,
            stroke='white',
            stroke_width=2
        )
        
        # Create path animation
        if len(waypoints) > 1:
            path_data_str = f"M{waypoints[0]['x']},{waypoints[0]['y']}"
            for point in waypoints[1:]:
                path_data_str += f" L{point['x']},{point['y']}"
            
            # Add path visualization
            dwg.add(dwg.path(
                d=path_data_str,
                stroke=color,
                stroke_width=3,
                stroke_dasharray="5,5",
                fill='none',
                opacity=0.5
            ))
            
            # Add motion animation
            animate_motion = dwg.animateMotion(
                path=path_data_str,
                dur="8s",
                repeatCount="1",
                fill="freeze"
            )
            vehicle.add(animate_motion)
        
        dwg.add(vehicle)
        
        # Add vehicle label
        dwg.add(dwg.text(
            label,
            insert=(start_point['x'] - 40, start_point['y'] - 25),
            fill=color,
            font_size=14,
            font_weight='bold'
        ))

    # 🆕 ADD: Missing methods that your existing code calls
    def load_full_route_data(self) -> dict:
        """Load the full route data with waypoints"""
        try:
            full_key = f"animation-data/{self.connection_id}/complete_route_data.json"
            response = self.s3.get_object(Bucket=self.bucket, Key=full_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"⚠️ Could not load full route data: {e}")
            return {}

    def extract_waypoints_for_lane(self, lane_id: str, full_data: dict) -> list:
        """Extract waypoints for a specific lane"""
        try:
            # Look in full route data or lane tree data
            if 'lane_trees' in full_data:
                lane_trees = full_data['lane_trees']
            else:
                # Load lane tree data
                self.load_lane_tree_data()
                lane_trees = self.lane_tree_data.get('lane_trees', []) if self.lane_tree_data else []
            
            for lane in lane_trees:
                if str(lane.get('lane_id', '')) == str(lane_id):
                    points = lane.get('points', [])
                    waypoints = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            waypoints.append({'x': float(point[0]), 'y': float(point[1])})
                        elif isinstance(point, dict) and 'x' in point and 'y' in point:
                            waypoints.append({'x': float(point['x']), 'y': float(point['y'])})
                    return waypoints
            
            return []
        except Exception as e:
            logger.warning(f"⚠️ Could not extract waypoints for lane {lane_id}: {e}")
            return []

    def convert_waypoints_to_svg(self, waypoints: list) -> list:
        """Convert waypoints to SVG coordinate system"""
        if not waypoints:
            return [{'x': 400, 'y': 300}, {'x': 600, 'y': 400}]
        
        svg_waypoints = []
        for point in waypoints:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                # Simple transformation to SVG canvas
                svg_x = max(50, min(self.svg_width - 50, point['x'] * 0.8))
                svg_y = max(50, min(self.svg_height - 50, point['y'] * 0.8))
                svg_waypoints.append({'x': svg_x, 'y': svg_y})
        
        return svg_waypoints if svg_waypoints else [{'x': 400, 'y': 300}]

    def add_intersection_background(self, dwg):
        """Add intersection background"""
        # Simple intersection
        dwg.add(dwg.line(
            start=(0, self.svg_height//2),
            end=(self.svg_width, self.svg_height//2),
            stroke='white',
            stroke_width=4
        ))
        dwg.add(dwg.line(
            start=(self.svg_width//2, 0),
            end=(self.svg_width//2, self.svg_height),
            stroke='white',
            stroke_width=4
        ))

    def add_collision_animation(self, dwg, collision_data: dict):
        """Add collision animation effect"""
        center_x = self.svg_width // 2
        center_y = self.svg_height // 2
        
        explosion = dwg.circle(
            center=(center_x, center_y),
            r=5,
            fill='orange',
            opacity=0
        )
        
        explosion.add(dwg.animate(
            attributeName='r',
            values='5;50;5',
            dur='1s',
            begin='6s',
            repeatCount='1'
        ))
        
        explosion.add(dwg.animate(
            attributeName='opacity',
            values='0;1;0',
            dur='1s',
            begin='6s',
            repeatCount='1'
        ))
        
        dwg.add(explosion)

    def add_enhanced_timeline(self, dwg, animation_guidance: dict):
        """Add enhanced timeline"""
        timeline_bg = dwg.rect(
            insert=(50, self.svg_height - 100),
            size=(self.svg_width - 100, 80),
            fill='black',
            opacity=0.8,
            rx=10
        )
        dwg.add(timeline_bg)
        
        dwg.add(dwg.text(
            "Two-Vehicle Collision Animation",
            insert=(self.svg_width//2, self.svg_height - 60),
            text_anchor="middle",
            fill='white',
            font_size=16
        ))

    def create_fallback_analysis(self, llm_data: dict) -> dict:
        """Create fallback analysis when LLM fails"""
        return {
            "vehicle_1": {
                "best_lane_id": "fallback_1",
                "start_position": "approach_from_south",
                "path_description": "fallback",
                "maneuver": "straight",
                "collision_timing": "60%"
            },
            "vehicle_2": {
                "best_lane_id": "fallback_2",
                "start_position": "approach_from_east",
                "path_description": "fallback",
                "maneuver": "straight",
                "collision_timing": "65%"
            },
            "collision": {"type": "intersection_collision"},
            "animation_guidance": {"total_duration": "8s", "key_moments": []}
        }

    def create_fallback_animation(self) -> str:
        """Create fallback animation using basic generator"""
        try:
            basic_generator = SVGAnimationGenerator(self.bucket, self.connection_id)
            fallback_route = {
                "vehicles": {
                    "user_vehicle": {
                        "path": {
                            "origin": {"direction": "south", "road_name": "Main St"},
                            "intended_destination": {"maneuver": "straight"}
                        }
                    },
                    "other_vehicle": {
                        "path": {
                            "origin": {"direction": "east", "road_name": "Oak Ave"},
                            "intended_destination": {"maneuver": "straight"}
                        }
                    }
                }
            }
            return basic_generator.generate_accident_animation(fallback_route)
        except Exception as e:
            logger.error(f"❗ Fallback failed: {e}")
            return self.create_minimal_svg()

    def create_minimal_svg(self) -> str:
        """Ultimate fallback"""
        dwg = svgwrite.Drawing(size=(800, 600))
        dwg.add(dwg.rect(insert=(0, 0), size=(800, 600), fill='lightgray'))
        dwg.add(dwg.text(
            "Animation generation failed",
            insert=(400, 300),
            text_anchor="middle",
            fill='red',
            font_size=20
        ))
        
        svg_key = f"animations/{self.connection_id}/error_animation.svg"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=svg_key,
            Body=dwg.tostring(),
            ContentType='image/svg+xml'
        )
        return svg_key

    def load_lane_tree_data(self):
        """Load lane tree data for analytics"""
        try:
            lane_tree_key = f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"
            response = self.s3.get_object(Bucket=self.bucket, Key=lane_tree_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"✅ Loaded lane tree data for analytics")
        except Exception as e:
            logger.error(f"❗ Failed to load lane tree: {e}")
            self.lane_tree_data = None

    # 🆕 ADD: Analytics-specific methods
    def extract_analytics_vehicle_path(self, llm_data: dict, vehicle_type: str) -> dict:
        """Extract vehicle path from analytics data"""
        accident_scenario = llm_data.get('accident_scenario', {})
        vehicle_data = accident_scenario.get(f'{vehicle_type}_vehicle', {})
        
        if not vehicle_data:
            return {'waypoints': [{'x': 400, 'y': 300}], 'color': 'gray', 'label': f'{vehicle_type} (no data)'}
        
        # Use route_points from analytics if available
        route_points = vehicle_data.get('path', {}).get('route_points', [])
        if route_points:
            waypoints = []
            for point in route_points:
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                    waypoints.append(point)
                elif isinstance(point, dict) and 'coordinates' in point:
                    coords = point['coordinates']
                    waypoints.append({'x': coords.get('x', 0), 'y': coords.get('y', 0)})
        else:
            waypoints = [{'x': 400, 'y': 300}, {'x': 600, 'y': 400}]
        
        svg_waypoints = self.convert_waypoints_to_svg(waypoints)
        
        return {
            'waypoints': svg_waypoints,
            'color': 'blue' if vehicle_type == 'user' else 'red',
            'label': 'Your Vehicle' if vehicle_type == 'user' else 'Other Vehicle',
            'confidence': 'analytics_precise'
        }

    def create_analytics_svg(self, user_path: dict, other_path: dict, llm_data: dict) -> str:
        """Create SVG using analytics guidance"""
        dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
        
        self.add_intersection_background(dwg)
        self.add_analytics_vehicle_animation(dwg, user_path)
        self.add_analytics_vehicle_animation(dwg, other_path)
        
        # Add collision point
        center_x, center_y = self.svg_width // 2, self.svg_height // 2
        dwg.add(dwg.circle(center=(center_x, center_y), r=10, fill='orange', opacity=0.7))
        dwg.add(dwg.text("COLLISION", insert=(center_x-30, center_y+30), fill='red', font_size=12))
        
        return dwg.tostring()

    def add_analytics_vehicle_animation(self, dwg, path_data: dict):
        """Add vehicle animation using analytics data"""
        waypoints = path_data['waypoints']
        if not waypoints:
            return
        
        # Vehicle shape
        vehicle = dwg.circle(
            center=(waypoints[0]['x'], waypoints[0]['y']),
            r=10,
            fill=path_data['color'],
            stroke='white',
            stroke_width=2
        )
        
        # Path
        if len(waypoints) > 1:
            path_data_str = f"M{waypoints[0]['x']},{waypoints[0]['y']}"
            for point in waypoints[1:]:
                path_data_str += f" L{point['x']},{point['y']}"
            
            # Animation
            animate_motion = dwg.animateMotion(
                path=path_data_str,
                dur="8s",
                repeatCount="indefinite"
            )
            vehicle.add(animate_motion)
            
            # Path visualization
            dwg.add(dwg.path(
                d=path_data_str,
                stroke=path_data['color'],
                stroke_width=3,
                fill='none',
                opacity=0.5
            ))
        
        dwg.add(vehicle)
        
        # Label
        dwg.add(dwg.text(
            path_data['label'],
            insert=(waypoints[0]['x'] - 30, waypoints[0]['y'] - 20),
            fill=path_data['color'],
            font_size=12
        ))

# 🚀 MAIN PROCESS
if __name__ == "__main__":
    try:
        bucket_name = _get_env("BUCKET_NAME")
        connection_id = _get_env("CONNECTION_ID")
        mode = os.environ.get("MODE", "ENHANCED_LLM_GUIDED")
        
        logger.info(f"🚀 Starting SVG Animation Generation")
        logger.info(f"🪣 Bucket: {bucket_name}")
        logger.info(f"🔗 Connection: {connection_id}")
        logger.info(f"🎯 Mode: {mode}")
        
        generator = EnhancedSVGAnimationGenerator(bucket_name, connection_id)
        svg_key = generator.generate_intelligent_animation()
        
        result = {
            "statusCode": 200,
            "svg_key": svg_key,
            "mode": mode,
            "connection_id": connection_id
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"❗ Main process failed: {e}")
        print(json.dumps({"statusCode": 500, "error": str(e)}, indent=2))
        sys.exit(1)