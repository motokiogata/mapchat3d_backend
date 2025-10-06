#svg_animation_generator.py
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
        
    def generate_accident_animation(self, route_json: Dict) -> str:
        """Generate basic accident animation"""
        try:
            logger.info("🎬 Generating basic accident animation")
            
            # Create simple two-vehicle animation
            dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
            
            # Background
            dwg.add(dwg.rect(insert=(0, 0), size=(self.svg_width, self.svg_height), fill='#1a1a1a'))
            
            # Simple intersection
            dwg.add(dwg.line(start=(0, self.svg_height//2), end=(self.svg_width, self.svg_height//2), stroke='white', stroke_width=4))
            dwg.add(dwg.line(start=(self.svg_width//2, 0), end=(self.svg_width//2, self.svg_height), stroke='white', stroke_width=4))
            
            # Vehicle 1 (horizontal movement)
            vehicle1 = dwg.circle(center=(100, self.svg_height//2), r=12, fill='blue', stroke='white', stroke_width=2)
            animate1 = dwg.animateMotion(path=f"M100,{self.svg_height//2} L{self.svg_width-100},{self.svg_height//2}", dur="8s", repeatCount="1")
            vehicle1.add(animate1)
            dwg.add(vehicle1)
            
            # Vehicle 2 (vertical movement)
            vehicle2 = dwg.circle(center=(self.svg_width//2, 100), r=12, fill='red', stroke='white', stroke_width=2)
            animate2 = dwg.animateMotion(path=f"M{self.svg_width//2},100 L{self.svg_width//2},{self.svg_height-100}", dur="8s", repeatCount="1")
            vehicle2.add(animate2)
            dwg.add(vehicle2)
            
            # Collision effect
            explosion = dwg.circle(center=(self.svg_width//2, self.svg_height//2), r=5, fill='orange', opacity=0)
            explosion.add(dwg.animate(attributeName='r', values='5;50;30', dur='2s', begin='4s', repeatCount='1'))
            explosion.add(dwg.animate(attributeName='opacity', values='0;1;0.5;0', dur='2s', begin='4s', repeatCount='1'))
            dwg.add(explosion)
            
            # Save to S3
            svg_key = f"animations/{self.connection_id}/basic_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=dwg.tostring(),
                ContentType='image/svg+xml'
            )
            
            logger.info(f"✅ Basic animation saved: {svg_key}")
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ Basic animation failed: {e}")
            return self.create_error_animation()
    
    def create_error_animation(self) -> str:
        """Create error animation"""
        dwg = svgwrite.Drawing(size=(800, 600))
        dwg.add(dwg.rect(insert=(0, 0), size=(800, 600), fill='lightgray'))
        dwg.add(dwg.text("Animation Error", insert=(400, 300), text_anchor="middle", fill='red', font_size=20))
        
        svg_key = f"animations/{self.connection_id}/error_animation.svg"
        self.s3.put_object(Bucket=self.bucket, Key=svg_key, Body=dwg.tostring(), ContentType='image/svg+xml')
        return svg_key

    def load_lane_tree_data(self):
        """Load lane tree data"""
        try:
            lane_tree_key = f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"
            response = self.s3.get_object(Bucket=self.bucket, Key=lane_tree_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"✅ Loaded lane tree data")
        except Exception as e:
            logger.error(f"❗ Failed to load lane tree: {e}")
            self.lane_tree_data = None

# 🔧 FIX: EnhancedSVGAnimationGenerator (FIXED NULL CHECKS)
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
        self.use_llm_validation = True  # ✅ ADD THIS LINE


    def create_default_scenario_analysis(self) -> dict:
        """Fallback scenario when extraction fails"""
        logger.warning("⚠️ Using default scenario analysis")
        return {
            "vehicle_1": {
                "best_lane_id": None,
                "maneuver": "straight",
                "collision_timing": "60%"
            },
            "vehicle_2": {
                "best_lane_id": None,
                "maneuver": "straight",
                "collision_timing": "65%"
            },
            "collision": {
                "type": "intersection_collision"
            }
        }

    def generate_guided_vehicle_paths(self, scenario_analysis: dict, full_data: dict) -> dict:
        """Generate vehicle paths using LLM guidance and full waypoint data"""
        logger.info("🛣️ Generating guided vehicle paths")
        
        try:
            # 🔧 FIX: Add null checks
            if not scenario_analysis:
                logger.warning("⚠️ No scenario analysis provided")
                return self.create_fallback_vehicle_paths()
            
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
                # Extract coordinates
                if isinstance(p1, dict):
                    p1_coords = [p1.get('x', 0), p1.get('y', 0)]
                else:
                    p1_coords = p1
                
                if isinstance(p2, dict):
                    p2_coords = [p2.get('x', 0), p2.get('y', 0)]
                else:
                    p2_coords = p2
                
                distance = math.sqrt((p1_coords[0] - p2_coords[0])**2 + (p1_coords[1] - p2_coords[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    # Average of closest points
                    intersection_point = [(p1_coords[0] + p2_coords[0]) / 2, (p1_coords[1] + p2_coords[1]) / 2]
        
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


    def generate_analytics_guided_animation(self) -> str:
        """Enhanced: Generate dual-vehicle animation with FULL DEBUGGING"""
        logger.info("🎯 DEBUG: Starting enhanced dual-vehicle analytics-guided animation")
        
        try:
            # 🔧 DEBUG: Safe data loading with detailed logging
            logger.info("🔍 DEBUG: Step 1 - Loading LLM data...")
            llm_data = self.load_llm_ready_data()
            
            if not llm_data:
                logger.warning("⚠️ DEBUG: No LLM data found, creating fallback")
                return self.create_fallback_animation()
            
            logger.info("🔍 DEBUG: Step 2 - Loading lane tree data...")
            self.load_lane_tree_data()
            
            # Extract accident scenario with null checks
            logger.info("🔍 DEBUG: Step 3 - Extracting user path...")
            user_path_data = llm_data.get('user_path', {})
            if not user_path_data:
                logger.warning("⚠️ DEBUG: No user_path found in LLM data, using fallback")
                return self.create_fallback_animation()
            
            # Get the approach lane
            approach_lane = user_path_data.get('approach_lane', '')
            logger.info(f"🔍 DEBUG: Target approach_lane = '{approach_lane}'")
            
            if not approach_lane:
                logger.warning("⚠️ DEBUG: No approach_lane specified, using fallback")
                return self.create_fallback_animation()
            
            # Extract waypoints
            logger.info("🔍 DEBUG: Step 4 - Extracting waypoints...")
            user_waypoints = self.extract_waypoints_for_lane(
                approach_lane, 
                {'lane_trees': self.lane_tree_data.get('lane_trees', []) if self.lane_tree_data else []}
            )
            
            logger.info(f"🔍 DEBUG: Got {len(user_waypoints)} waypoints for user vehicle")
            
            if not user_waypoints:
                logger.warning("⚠️ DEBUG: No waypoints extracted, using fallback")
                return self.create_fallback_animation()
            
            # Create paths
            logger.info("🔍 DEBUG: Step 5 - Creating vehicle paths...")
            user_path = {
                'waypoints': self.convert_waypoints_to_svg(user_waypoints),
                'color': 'blue',
                'label': f'Your Vehicle ({user_path_data.get("intended_maneuver", "unknown")})',
                'vehicle_type': 'user'
            }
            
            # Create other vehicle path (simplified for now)
            other_path = {
                'waypoints': self.create_fallback_waypoints('other'),
                'color': 'red', 
                'label': 'Other Vehicle',
                'vehicle_type': 'other'
            }
            
            # Calculate collision timing and point
            logger.info("🔍 DEBUG: Step 6 - Calculating collision...")
            collision_data = self.calculate_collision_timing(user_path, other_path)
            
            # Create dual-vehicle SVG animation
            logger.info("🔍 DEBUG: Step 7 - Creating SVG animation...")
            svg_content = self.create_dual_vehicle_svg_animation(user_path, other_path, collision_data)
            
            # Save to S3
            svg_key = f"animations/{self.connection_id}/debug_analytics_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=svg_content,
                ContentType='image/svg+xml'
            )
            
            logger.info(f"🔍 DEBUG: ✅ SUCCESS! Animation saved: {svg_key}")
            logger.info(f"🔍 DEBUG: User waypoints: {len(user_path.get('waypoints', []))}")
            logger.info(f"🔍 DEBUG: Other waypoints: {len(other_path.get('waypoints', []))}")
            
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ DEBUG: Enhanced animation failed: {e}")
            import traceback
            logger.error(f"❗ DEBUG: Full traceback: {traceback.format_exc()}")
            return self.create_fallback_animation()


    # 🔧 FIX: Load data with proper error handling
    def load_llm_ready_data(self) -> dict:
        """Load the lightweight metadata-only data with ENHANCED DEBUGGING"""
        try:
            llm_key = f"animation-data/{self.connection_id}/llm_ready_data.json"
            logger.info(f"🔍 DEBUG: Looking for LLM data at: {llm_key}")
            
            response = self.s3.get_object(Bucket=self.bucket, Key=llm_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"🔍 DEBUG: LLM data loaded successfully")
            logger.info(f"🔍 DEBUG: LLM data keys: {list(data.keys()) if data else 'EMPTY'}")
            
            if 'user_path' in data:
                user_path = data['user_path']
                logger.info(f"🔍 DEBUG: user_path keys: {list(user_path.keys())}")
                logger.info(f"🔍 DEBUG: approach_lane = '{user_path.get('approach_lane', 'NOT_FOUND')}'")
                logger.info(f"🔍 DEBUG: origin_road_id = '{user_path.get('origin_road_id', 'NOT_FOUND')}'")
            else:
                logger.warning(f"🔍 DEBUG: No 'user_path' found in LLM data!")
            
            return data
            
        except Exception as e:
            logger.error(f"❗ DEBUG: Failed to load LLM data: {e}")
            logger.error(f"❗ DEBUG: Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❗ DEBUG: Full traceback: {traceback.format_exc()}")
            return {}



    def generate_llm_guided_animation(self) -> str:
        """LLM-guided method with better error handling"""
        try:
            # 1. Load LLM-ready data (metadata only)
            llm_data = self.load_llm_ready_data()
            
            if not llm_data:
                return self.create_fallback_animation()
            
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
            
        except Exception as e:
            logger.error(f"❗ LLM guided animation failed: {e}")
            return self.create_fallback_animation()

    ############### Helper Methods ###############

    def find_matching_lane(self, approach_lane: str, available_lanes: list) -> str:
        """Match approach_lane description to actual lane_id"""
        
        # Direct match first
        if approach_lane in available_lanes:
            return approach_lane
        
        # Fuzzy matching (e.g., "SE" matches "road_X_SE_lane")
        for lane_id in available_lanes:
            if approach_lane.upper() in lane_id.upper():
                logger.info(f"✅ Matched '{approach_lane}' → '{lane_id}'")
                return lane_id
        
        # Extract direction and match
        direction_map = {
            'SOUTH': 'S', 'NORTH': 'N', 'EAST': 'E', 'WEST': 'W',
            'SOUTHEAST': 'SE', 'NORTHWEST': 'NW', 'NORTHEAST': 'NE', 'SOUTHWEST': 'SW'
        }
        
        for direction, abbrev in direction_map.items():
            if direction in approach_lane.upper():
                for lane_id in available_lanes:
                    if abbrev in lane_id:
                        logger.info(f"✅ Direction matched '{approach_lane}' → '{lane_id}'")
                        return lane_id
        
        # Fallback to first available lane
        logger.warning(f"⚠️ No match for '{approach_lane}', using first available")
        return available_lanes[0] if available_lanes else None


    def find_conflicting_lane(self, vehicle_1_lane: str, available_lanes: list) -> str:
        """Find a lane that would intersect with vehicle_1's lane - ENHANCED"""
        
        if not vehicle_1_lane or not available_lanes:
            return available_lanes[1] if len(available_lanes) > 1 else available_lanes[0] if available_lanes else "road_1_NE_lane"
        
        logger.info(f"🔍 Finding conflicting lane for: {vehicle_1_lane}")
        
        # Extract direction from vehicle 1's lane
        v1_direction = self.extract_direction(vehicle_1_lane)
        logger.info(f"🔍 Vehicle 1 direction: {v1_direction}")
        
        # Find perpendicular lane
        perpendicular_dirs = {
            'SE': ['NE', 'SW', 'NW'], 
            'NW': ['NE', 'SW', 'SE'],
            'NE': ['SE', 'NW', 'SW'], 
            'SW': ['SE', 'NW', 'NE'],
            'S': ['E', 'W', 'N'],
            'N': ['E', 'W', 'S'],
            'E': ['N', 'S', 'W'],
            'W': ['N', 'S', 'E']
        }
        
        target_dirs = perpendicular_dirs.get(v1_direction, ['NE', 'SW', 'NW'])
        logger.info(f"🔍 Looking for conflicting directions: {target_dirs}")
        
        # Search for conflicting lanes
        for lane_id in available_lanes:
            if lane_id != vehicle_1_lane:
                lane_dir = self.extract_direction(lane_id)
                logger.info(f"🔍 Checking lane {lane_id} with direction {lane_dir}")
                if lane_dir in target_dirs:
                    logger.info(f"✅ Found conflicting lane: {lane_id}")
                    return lane_id
        
        # Fallback to any different lane
        for lane_id in available_lanes:
            if lane_id != vehicle_1_lane:
                logger.info(f"✅ Fallback conflicting lane: {lane_id}")
                return lane_id
        
        # Ultimate fallback
        logger.warning("⚠️ No conflicting lane found, using fallback")
        return "road_1_NE_lane"

    def extract_direction(self, lane_id: str) -> str:
        """Extract direction abbreviation from lane_id"""
        directions = ['SE', 'NW', 'NE', 'SW', 'N', 'S', 'E', 'W']
        for direction in directions:
            if direction in lane_id.upper():
                return direction
        return ''


    def call_bedrock_for_analysis(self, llm_data: dict, vehicle_1_lane: str, vehicle_2_lane: str) -> dict:
        """Use AWS Bedrock LLM to validate and enhance collision scenario analysis"""
        
        logger.info("🤖 Calling Bedrock for scenario validation...")
        
        try:
            # Prepare the prompt for the LLM
            prompt = f"""
            Analyze this vehicle collision scenario and provide movement details:

            USER VEHICLE:
            - Lane: {vehicle_1_lane}
            - Path: {llm_data.get('user_path', {})}

            OTHER VEHICLE:
            - Lane: {vehicle_2_lane}

            Return JSON with collision analysis:
            {{
                "vehicle_1": {{
                    "best_lane_id": "{vehicle_1_lane}",
                    "maneuver": "straight",
                    "collision_timing": "60%"
                }},
                "vehicle_2": {{
                    "best_lane_id": "{vehicle_2_lane}",
                    "maneuver": "straight", 
                    "collision_timing": "65%"
                }},
                "collision": {{
                    "type": "intersection_collision"
                }}
            }}
            """

            # Bedrock request - USE INFERENCE PROFILE
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }

            # 🔧 FIX: Use the inference profile from your template.yaml
            response = self.bedrock.invoke_model(
                modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",  # This matches your IAM policy
                body=json.dumps(request_body),
                contentType="application/json"
            )

            # Parse response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            llm_text = response_body['content'][0]['text']
            
            # Extract JSON from response
            start = llm_text.find('{')
            end = llm_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = llm_text[start:end]
                return json.loads(json_str)
            else:
                logger.warning("⚠️ No JSON found in LLM response")
                return self.create_default_scenario_analysis()

        except Exception as e:
            logger.error(f"❗ Bedrock call failed: {e}")
            return self.create_default_scenario_analysis()
            


    def analyze_scenario_with_llm(self, llm_data: dict) -> dict:
        """Analyze scenario with LLM - FIXED IMPLEMENTATION"""
        
        logger.info("🤖 Analyzing scenario with LLM...")
        
        try:
            # 1. Extract actual data from llm_data
            user_path = llm_data.get('user_path', {})
            approach_lane = user_path.get('approach_lane', '')
            origin_road_id = user_path.get('origin_road_id', '')
            
            logger.info(f"🔍 DEBUG: approach_lane from llm_data: '{approach_lane}'")
            logger.info(f"🔍 DEBUG: origin_road_id from llm_data: '{origin_road_id}'")
            
            # 2. Load available lanes
            if not self.lane_tree_data:
                self.load_lane_tree_data()
            
            available_lanes = [
                lane.get('lane_id') 
                for lane in self.lane_tree_data.get('lane_trees', [])
                if lane.get('lane_id')  # 🔧 FIX: Only include lanes with valid IDs
            ] if self.lane_tree_data else []
            
            logger.info(f"📋 Available lanes: {available_lanes}")
            logger.info(f"🎯 Target approach_lane: '{approach_lane}'")
            
            # 3. Match approach_lane to actual lane_id
            if approach_lane and approach_lane != "unknown" and approach_lane != "None":
                vehicle_1_lane = approach_lane  # Use it directly if it's valid
                if vehicle_1_lane not in available_lanes:
                    logger.warning(f"⚠️ Lane '{vehicle_1_lane}' not in available lanes, trying to match...")
                    vehicle_1_lane = self.find_matching_lane(approach_lane, available_lanes)
            else:
                logger.warning(f"⚠️ Invalid approach_lane: '{approach_lane}', using first available")
                vehicle_1_lane = available_lanes[0] if available_lanes else "road_0_SE_lane"
            
            # 4. Find conflicting lane for vehicle 2
            vehicle_2_lane = self.find_conflicting_lane(vehicle_1_lane, available_lanes)
            
            logger.info(f"✅ Final lane assignment: vehicle_1={vehicle_1_lane}, vehicle_2={vehicle_2_lane}")
            
            # 5. (Optional) Use Bedrock LLM to validate/enhance
            if self.use_llm_validation:
                return self.call_bedrock_for_analysis(llm_data, vehicle_1_lane, vehicle_2_lane)
            
            # 6. Return actual analysis
            return {
                "vehicle_1": {
                    "best_lane_id": vehicle_1_lane,
                    "maneuver": user_path.get('intended_maneuver', 'straight'),
                    "collision_timing": "60%"
                },
                "vehicle_2": {
                    "best_lane_id": vehicle_2_lane,
                    "maneuver": "straight",
                    "collision_timing": "65%"
                },
                "collision": {
                    "type": "intersection_collision"
                }
            }
            
        except Exception as e:
            logger.error(f"❗ LLM analysis failed: {e}")
            return self.create_default_scenario_analysis()

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
        if isinstance(start_point, dict):
            center = (start_point['x'], start_point['y'])
        else:
            center = (start_point[0], start_point[1])
        
        vehicle = dwg.circle(
            center=center,
            r=12,
            fill=color,
            stroke='white',
            stroke_width=2
        )
        
        # Create path animation
        if len(waypoints) > 1:
            path_data_str = self.waypoints_to_svg_path(waypoints)
            
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
            insert=(center[0] - 40, center[1] - 25),
            fill=color,
            font_size=14,
            font_weight='bold'
        ))

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
        """Extract waypoints for a specific lane with ENHANCED DEBUGGING"""
        logger.info(f"🔍 DEBUG: extract_waypoints_for_lane called with lane_id='{lane_id}'")
        
        try:
            # Check what full_data contains
            logger.info(f"🔍 DEBUG: full_data keys: {list(full_data.keys()) if full_data else 'EMPTY'}")
            
            # Look in full route data or lane tree data
            if 'lane_trees' in full_data:
                lane_trees = full_data['lane_trees']
                logger.info(f"🔍 DEBUG: Using lane_trees from full_data, count: {len(lane_trees)}")
            else:
                # Load lane tree data
                logger.info(f"🔍 DEBUG: No lane_trees in full_data, loading from lane_tree_data")
                if not self.lane_tree_data:
                    logger.info(f"🔍 DEBUG: lane_tree_data is None, calling load_lane_tree_data()")
                    self.load_lane_tree_data()
                
                lane_trees = self.lane_tree_data.get('lane_trees', []) if self.lane_tree_data else []
                logger.info(f"🔍 DEBUG: Using lane_trees from self.lane_tree_data, count: {len(lane_trees)}")
            
            if not lane_trees:
                logger.warning(f"🔍 DEBUG: No lane_trees available!")
                return []
            
            # Search for matching lane
            logger.info(f"🔍 DEBUG: Searching for lane_id '{lane_id}' in {len(lane_trees)} lanes")
            
            for i, lane in enumerate(lane_trees):
                current_lane_id = lane.get('lane_id', 'NO_ID')
                logger.info(f"🔍 DEBUG: Checking lane {i}: '{current_lane_id}' vs target '{lane_id}'")
                
                # Try exact match first
                if current_lane_id == lane_id:
                    logger.info(f"🔍 DEBUG: ✅ EXACT MATCH found for lane_id '{lane_id}'!")
                    
                    points = lane.get('points', [])
                    logger.info(f"🔍 DEBUG: Lane has {len(points)} points")
                    
                    if points:
                        logger.info(f"🔍 DEBUG: First point: {points[0] if points else 'NONE'}")
                        logger.info(f"🔍 DEBUG: Point type: {type(points[0]) if points else 'NONE'}")
                    
                    waypoints = []
                    for j, point in enumerate(points):
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            waypoint = {'x': float(point[0]), 'y': float(point[1])}
                            waypoints.append(waypoint)
                            if j < 3:  # Log first 3 waypoints
                                logger.info(f"🔍 DEBUG: Waypoint {j}: {waypoint}")
                        elif isinstance(point, dict) and 'x' in point and 'y' in point:
                            waypoint = {'x': float(point['x']), 'y': float(point['y'])}
                            waypoints.append(waypoint)
                            if j < 3:  # Log first 3 waypoints
                                logger.info(f"🔍 DEBUG: Waypoint {j}: {waypoint}")
                        else:
                            logger.warning(f"🔍 DEBUG: Invalid point format at index {j}: {point}")
                    
                    logger.info(f"🔍 DEBUG: Successfully extracted {len(waypoints)} waypoints")
                    return waypoints
                
                # Try string comparison
                elif str(current_lane_id) == str(lane_id):
                    logger.info(f"🔍 DEBUG: ✅ STRING MATCH found for lane_id '{lane_id}'!")
                    # Same logic as above...
                    points = lane.get('points', [])
                    waypoints = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            waypoints.append({'x': float(point[0]), 'y': float(point[1])})
                        elif isinstance(point, dict) and 'x' in point and 'y' in point:
                            waypoints.append({'x': float(point['x']), 'y': float(point['y'])})
                    return waypoints
            
            logger.warning(f"🔍 DEBUG: ❌ NO MATCH found for lane_id '{lane_id}'")
            logger.warning(f"🔍 DEBUG: Available lane_ids were: {[lane.get('lane_id', 'NO_ID') for lane in lane_trees]}")
            return []
            
        except Exception as e:
            logger.error(f"❗ DEBUG: extract_waypoints_for_lane failed: {e}")
            import traceback
            logger.error(f"❗ DEBUG: Full traceback: {traceback.format_exc()}")
            return []




    def convert_waypoints_to_svg(self, waypoints: list) -> list:
        """Convert waypoints to SVG coordinate system with DEBUGGING"""
        logger.info(f"🔍 DEBUG: convert_waypoints_to_svg called with {len(waypoints)} waypoints")
        
        if not waypoints:
            logger.warning("🔍 DEBUG: No waypoints provided, returning default")
            return [{'x': 400, 'y': 300}, {'x': 600, 'y': 400}]
        
        svg_waypoints = []
        for i, point in enumerate(waypoints):
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                # Simple transformation to SVG canvas (adjust as needed)
                original_x, original_y = point['x'], point['y']
                svg_x = max(50, min(self.svg_width - 50, original_x * 0.8 + 100))
                svg_y = max(50, min(self.svg_height - 50, original_y * 0.8 + 100))
                
                svg_waypoint = {'x': svg_x, 'y': svg_y}
                svg_waypoints.append(svg_waypoint)
                
                if i < 3:  # Log first 3 conversions
                    logger.info(f"🔍 DEBUG: Waypoint {i}: ({original_x}, {original_y}) → ({svg_x}, {svg_y})")
            else:
                logger.warning(f"🔍 DEBUG: Invalid waypoint format at index {i}: {point}")
        
        logger.info(f"🔍 DEBUG: Converted {len(svg_waypoints)} waypoints to SVG coordinates")
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

    def extract_vehicle_path_from_metadata(self, accident_scenario: dict, vehicle_type: str) -> dict:
        """Extract vehicle path using algorithmic lane determination"""
        logger.info(f"🚗 Extracting {vehicle_type} vehicle path")
        
        try:
            vehicle_data = accident_scenario.get(f'{vehicle_type}_vehicle', {})
            if not vehicle_data:
                return self.create_fallback_vehicle_path(vehicle_type)
            
            # Create basic path based on vehicle type
            if vehicle_type == 'user':
                waypoints = [
                    {'x': 100, 'y': self.svg_height // 2},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ]
                color = 'blue'
                label = 'Your Vehicle'
            else:
                waypoints = [
                    {'x': self.svg_width // 2, 'y': 100},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ]
                color = 'red'
                label = 'Other Vehicle'
            
            return {
                'waypoints': waypoints,
                'vehicle_type': vehicle_type,
                'color': color,
                'label': label,
                'path_quality': 'extracted_from_metadata'
            }
            
        except Exception as e:
            logger.error(f"❗ Error extracting path for {vehicle_type}: {e}")
            return self.create_fallback_vehicle_path(vehicle_type)

    def create_fallback_vehicle_path(self, vehicle_type: str) -> dict:
        """Create fallback path when extraction fails"""
        if vehicle_type == 'user':
            waypoints = [{'x': 100, 'y': self.svg_height // 2}, {'x': self.svg_width // 2, 'y': self.svg_height // 2}]
            color = 'blue'
            label = 'Your Vehicle (fallback)'
        else:
            waypoints = [{'x': self.svg_width // 2, 'y': 100}, {'x': self.svg_width // 2, 'y': self.svg_height // 2}]
            color = 'red'
            label = 'Other Vehicle (fallback)'
        
        return {
            'waypoints': waypoints,
            'vehicle_type': vehicle_type,
            'color': color,
            'label': label,
            'path_quality': 'fallback'
        }

    def create_fallback_animation(self) -> str:
        """Create fallback animation using basic generator"""
        try:
            logger.info("🔄 Creating fallback animation")
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
        """Ultimate fallback - creates simple SVG"""
        logger.info("🆘 Creating minimal fallback SVG")
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
        logger.info(f"✅ Minimal SVG saved: {svg_key}")
        return svg_key


    def load_lane_tree_data(self):
        """Load lane tree data with ENHANCED DEBUGGING"""
        try:
            lane_tree_key = f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"
            logger.info(f"🔍 DEBUG: Looking for lane tree at: {lane_tree_key}")
            
            response = self.s3.get_object(Bucket=self.bucket, Key=lane_tree_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"🔍 DEBUG: Lane tree loaded successfully")
            logger.info(f"🔍 DEBUG: Lane tree keys: {list(self.lane_tree_data.keys()) if self.lane_tree_data else 'EMPTY'}")
            
            if 'lane_trees' in self.lane_tree_data:
                lane_trees = self.lane_tree_data['lane_trees']
                logger.info(f"🔍 DEBUG: Found {len(lane_trees)} lanes in lane_trees")
                
                # Debug first few lane IDs
                for i, lane in enumerate(lane_trees[:3]):  # Show first 3 lanes
                    lane_id = lane.get('lane_id', 'NO_ID')
                    road_id = lane.get('road_id', 'NO_ROAD_ID')
                    points_count = len(lane.get('points', []))
                    logger.info(f"🔍 DEBUG: Lane {i}: lane_id='{lane_id}', road_id={road_id}, points={points_count}")
                
                # Show all available lane_ids
                all_lane_ids = [lane.get('lane_id', 'NO_ID') for lane in lane_trees]
                logger.info(f"🔍 DEBUG: All available lane_ids: {all_lane_ids}")
            else:
                logger.warning(f"🔍 DEBUG: No 'lane_trees' key found in lane tree data!")
                
        except Exception as e:
            logger.error(f"❗ DEBUG: Failed to load lane tree: {e}")
            logger.error(f"❗ DEBUG: Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❗ DEBUG: Full traceback: {traceback.format_exc()}")
            self.lane_tree_data = None

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