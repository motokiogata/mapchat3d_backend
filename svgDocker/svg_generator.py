# svg_generator.py
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -------------------------
# 🔧 Small helpers (module level, not in class)
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

class SVGAnimationGenerator:
    def __init__(self, s3_bucket: str, connection_id: str):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.connection_id = connection_id
        self.lane_tree_data = None
        self.base_map_bounds = None
        self.svg_width = 800
        self.svg_height = 600
        
    ##############
    # Main method
    ##############
    def generate_accident_animation(self, route_json: Dict) -> str:
        """Main method to generate animated SVG"""
        try:
            # DEBUG: Print actual JSON structure
            logger.info(f"🔍 Loaded JSON keys: {list(route_json.keys())}")
            logger.info(f"🔍 JSON structure preview: {json.dumps(route_json, indent=2)[:500]}...")
            
            # 1. Load lane tree data
            self.load_lane_tree_data()
            
            # 2. Parse route scenario FIRST (this will handle different JSON structures)
            scenario_data = self.parse_route_scenario(route_json)
            
            # 3. Validate that we have vehicle data
            if not scenario_data.get('user_vehicle') or not scenario_data.get('other_vehicle'):
                logger.error(f"❗ Missing vehicle data. Scenario data: {scenario_data}")
                raise ValueError("Missing vehicle data in route JSON")
            
            # 4. Match vehicles to lane paths (use parsed data instead of direct access)
            user_path = self.match_vehicle_to_lanes(
                scenario_data['user_vehicle'], 'user'
            )
            other_path = self.match_vehicle_to_lanes(
                scenario_data['other_vehicle'], 'other'
            )
            
            # 5. Generate collision scenario
            scenario = CollisionScenario(
                user_vehicle=user_path,
                other_vehicle=other_path,
                collision_point=self.calculate_collision_point(route_json),
                collision_type=scenario_data.get('collision', {}).get('type', 'unknown'),
                base_map_image=f"{self.connection_id}_intersection_map.png"
            )
            
            # 6. Create animated SVG
            svg_content = self.create_animated_svg(scenario)
            
            # 7. Save to S3
            svg_key = f"animations/{self.connection_id}/accident_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=svg_content,
                ContentType='image/svg+xml'
            )
            
            logger.info(f"✅ Generated SVG animation: {svg_key}")
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ Animation generation failed: {e}")
            raise


    def load_lane_tree_data(self):
        """Load the lane tree routes enhanced JSON file (S3 only)."""
        try:
            lane_tree_key = getattr(self, "lane_tree_key_override", None) \
                            or os.getenv("LANE_TREE_KEY") \
                            or f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"

            logger.info(f"📥 Loading lane_tree from s3://{self.bucket}/{lane_tree_key}")
            response = self.s3.get_object(Bucket=self.bucket, Key=lane_tree_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))

            logger.info(f"✅ Loaded lane tree data with {len(self.lane_tree_data['lane_trees'])} lanes")
        except Exception as e:
            logger.error(f"❗ Failed to load lane tree data from S3: {e}")
            raise

    def match_vehicle_to_lanes(self, vehicle_data: Dict, vehicle_type: str) -> VehiclePath:
        """Match vehicle route description to actual lane paths using LLM"""
        
        # Extract vehicle info
        origin_direction = vehicle_data['path']['origin']['direction']
        maneuver = vehicle_data['path']['intended_destination']['maneuver']
        origin_road = vehicle_data['path']['origin']['road_name']
        
        # Find matching lanes using intelligent matching
        start_lane = self.find_best_matching_lane(
            direction=origin_direction,
            road_name=origin_road,
            purpose="origin"
        )
        
        end_lane = self.find_best_matching_lane(
            direction=self.calculate_end_direction(origin_direction, maneuver),
            maneuver=maneuver,
            purpose="destination"
        )
        
        # Generate route points
        route_points = self.generate_vehicle_route_points(start_lane, end_lane, maneuver)
        
        return VehiclePath(
            vehicle_id=vehicle_type,
            start_lane_id=start_lane['lane_id'],
            end_lane_id=end_lane['lane_id'] if end_lane else None,
            maneuver=maneuver,
            route_points=route_points,
            collision_point=route_points[-1] if route_points else (0, 0),
            timeline=[]
        )

    def find_best_matching_lane(self, direction: str = None, road_name: str = None, 
                               maneuver: str = None, purpose: str = "origin") -> Dict:
        """Find the best matching lane using AI-powered matching"""
        
        candidates = []
        
        for lane in self.lane_tree_data['lane_trees']:
            score = 0
            metadata = lane.get('metadata', {})
            
            # Direction matching
            if direction and direction != "unknown":
                lane_direction = metadata.get('simple_direction', '').lower()
                traffic_direction = metadata.get('traffic_direction', '').lower()
                
                if direction.lower() in lane_direction or direction.lower() in traffic_direction:
                    score += 30
                
                # Check directional language
                direction_langs = metadata.get('narrative_directional', {}).get('user_direction_language', [])
                if any(direction.lower() in lang.lower() for lang in direction_langs):
                    score += 20
            
            # Road name matching
            if road_name and road_name != "unknown":
                parent_road_name = metadata.get('parent_road_name', '').lower()
                display_name = metadata.get('display_name', '').lower()
                
                if road_name.lower() in parent_road_name or road_name.lower() in display_name:
                    score += 25
            
            # Maneuver capability matching
            if maneuver:
                if maneuver == "left_turn" and metadata.get('can_turn_left'):
                    score += 15
                elif maneuver == "right_turn" and metadata.get('can_turn_right'):
                    score += 15
                elif maneuver == "straight" and metadata.get('can_go_straight'):
                    score += 15
            
            # Lane type preferences
            lane_type = metadata.get('lane_type', '')
            if 'main' in lane_type.lower() or 'through' in lane_type.lower():
                score += 10
            
            candidates.append((lane, score))
        
        # Sort by score and return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_lane, best_score = candidates[0]
            logger.info(f"🎯 Best lane match for {purpose}: {best_lane['lane_id']} (score: {best_score})")
            return best_lane
        else:
            # Fallback to first available lane
            logger.warning(f"⚠️ No good lane match found, using fallback")
            return self.lane_tree_data['lane_trees'][0] if self.lane_tree_data['lane_trees'] else {}

    
    def generate_vehicle_route_points(self, start_lane: Dict, end_lane: Dict, maneuver: str) -> List[Tuple[float, float]]:
        """Generate smooth route points from start lane to end lane"""
        
        route_points = []
        
        # Get start lane points
        start_points = start_lane.get('points', [])
        if not start_points:
            logger.warning("No start points found")
            return [(0, 0), (100, 100)]
        
        # Convert map coordinate points if available
        map_coords = start_lane.get('metadata', {}).get('map_coordinate_points', [])
        if map_coords:
            # 🔥 FIX: Handle different coordinate formats
            logger.info(f"🔍 map_coords format check: {type(map_coords[0]) if map_coords else 'empty'}")
            
            try:
                # Check the format of coordinates
                if map_coords and isinstance(map_coords[0], dict):
                    # Format: [{'x': 640, 'y': 1180}, ...]
                    start_points = [(p['x'], p['y']) for p in map_coords[:len(start_points)//2]]
                elif map_coords and isinstance(map_coords[0], (list, tuple)) and len(map_coords[0]) >= 2:
                    # Format: [[640, 1180], ...] or [(640, 1180), ...]
                    start_points = [(p[0], p[1]) for p in map_coords[:len(start_points)//2]]
                elif map_coords and isinstance(map_coords[0], (int, float)):
                    # Format: [640, 1180, 640, 1160, ...] (flat list)
                    paired_coords = [(map_coords[i], map_coords[i+1]) 
                                for i in range(0, min(len(map_coords)-1, len(start_points)), 2)]
                    start_points = paired_coords
                else:
                    logger.warning(f"⚠️ Unknown map_coords format: {type(map_coords[0]) if map_coords else 'empty'}")
                    # Fallback to original start_points
                    start_points = start_points[:len(start_points)//2]
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"⚠️ Error processing map_coords: {e}, using original points")
                # Fallback to original start_points
                start_points = start_points[:len(start_points)//2]
        else:
            # Use lane tree points directly
            start_points = start_points[:len(start_points)//2]  # Take first half for approach
        
        # 🔥 FIX: Ensure start_points are tuples of floats
        processed_start_points = []
        for point in start_points:
            try:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    processed_start_points.append((float(point[0]), float(point[1])))
                elif isinstance(point, dict) and 'x' in point and 'y' in point:
                    processed_start_points.append((float(point['x']), float(point['y'])))
                else:
                    logger.warning(f"⚠️ Skipping invalid point: {point}")
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"⚠️ Error processing point {point}: {e}")
        
        if processed_start_points:
            route_points.extend(processed_start_points)
        else:
            # Ultimate fallback
            route_points = [(0, 0), (50, 50), (100, 100)]
            logger.warning("⚠️ Using fallback route points")
        
        # Add intersection transition points
        if end_lane and maneuver != "straight":
            try:
                # Generate turning curve
                end_points = end_lane.get('points', [])
                
                # Process end points the same way
                if end_points:
                    end_map_coords = end_lane.get('metadata', {}).get('map_coordinate_points', [])
                    if end_map_coords:
                        try:
                            if isinstance(end_map_coords[0], dict):
                                end_processed = [(p['x'], p['y']) for p in end_map_coords[:3]]
                            elif isinstance(end_map_coords[0], (list, tuple)):
                                end_processed = [(p[0], p[1]) for p in end_map_coords[:3]]
                            else:
                                end_processed = end_points[:3]
                        except (IndexError, KeyError, TypeError):
                            end_processed = end_points[:3]
                    else:
                        end_processed = end_points[:3]
                else:
                    end_processed = [(100, 100)]
                
                turn_points = self.generate_turn_curve(
                    route_points[-1] if route_points else (0, 0),
                    end_processed,
                    maneuver
                )
                route_points.extend(turn_points)
            except Exception as e:
                logger.warning(f"⚠️ Error generating turn curve: {e}")
        
        # Smooth the route
        try:
            smoothed_points = self.smooth_route_points(route_points)
            # 🔥 FIX: Transform to SVG coordinates
            svg_route_points = self.transform_all_route_points(smoothed_points)
            logger.info(f"🔄 Transformed {len(smoothed_points)} route points to SVG coordinates")
            return svg_route_points
        except Exception as e:
            logger.warning(f"⚠️ Error smoothing route: {e}")
            # 🔥 FIX: Transform fallback points too
            fallback_points = route_points if route_points else [(0, 0), (100, 100)]
            return self.transform_all_route_points(fallback_points)

            

    def generate_turn_curve(self, start_point: Tuple[float, float], 
                           end_points: List, maneuver: str) -> List[Tuple[float, float]]:
        """Generate smooth curve for turning maneuvers"""
        
        if not end_points:
            return [start_point, (start_point[0] + 50, start_point[1] + 50)]
        
        end_point = end_points[0] if isinstance(end_points[0], tuple) else (end_points[0], end_points[1])
        
        # Calculate control points for Bezier curve
        mid_x = (start_point[0] + end_point[0]) / 2
        mid_y = (start_point[1] + end_point[1]) / 2
        
        # Adjust control point based on turn type
        if maneuver == "left_turn":
            control_point = (mid_x - 20, mid_y - 20)
        elif maneuver == "right_turn":
            control_point = (mid_x + 20, mid_y + 20)
        else:
            control_point = (mid_x, mid_y)
        
        # Generate curve points
        curve_points = []
        for t in np.linspace(0, 1, 10):
            x = (1-t)**2 * start_point[0] + 2*(1-t)*t * control_point[0] + t**2 * end_point[0]
            y = (1-t)**2 * start_point[1] + 2*(1-t)*t * control_point[1] + t**2 * end_point[1]
            curve_points.append((x, y))
        
        return curve_points

    def smooth_route_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply smoothing to route points"""
        if len(points) < 3:
            return points
        
        # Simple moving average smoothing
        smoothed = [points[0]]  # Keep first point
        
        for i in range(1, len(points) - 1):
            prev_point = points[i-1]
            curr_point = points[i]
            next_point = points[i+1]
            
            smooth_x = (prev_point[0] + curr_point[0] + next_point[0]) / 3
            smooth_y = (prev_point[1] + curr_point[1] + next_point[1]) / 3
            
            smoothed.append((smooth_x, smooth_y))
        
        smoothed.append(points[-1])  # Keep last point
        return smoothed

    ############## 
    # SVG Generation
    ##############
    def transform_coordinates_to_svg(self, world_coords: Tuple[float, float]) -> Tuple[float, float]:
        """Transform world coordinates to SVG coordinate system"""
        
        if not self.lane_tree_data:
            return world_coords
        
        # Calculate bounds of the lane tree data
        all_x_coords = []
        all_y_coords = []
        
        for lane in self.lane_tree_data['lane_trees']:
            points = lane.get('points', [])
            for point in points:
                try:
                    if isinstance(point, (tuple, list)) and len(point) >= 2:
                        all_x_coords.append(float(point[0]))
                        all_y_coords.append(float(point[1]))
                    elif isinstance(point, dict) and 'x' in point and 'y' in point:
                        all_x_coords.append(float(point['x']))
                        all_y_coords.append(float(point['y']))
                except (ValueError, TypeError, IndexError):
                    continue
        
        if not all_x_coords or not all_y_coords:
            return world_coords
        
        # World bounds
        world_min_x, world_max_x = min(all_x_coords), max(all_x_coords)
        world_min_y, world_max_y = min(all_y_coords), max(all_y_coords)
        world_width = world_max_x - world_min_x
        world_height = world_max_y - world_min_y
        
        # Add padding
        padding = 50
        svg_width = self.svg_width - 2 * padding
        svg_height = self.svg_height - 2 * padding
        
        # Transform coordinates
        world_x, world_y = world_coords
        
        # Normalize to 0-1 range
        norm_x = (world_x - world_min_x) / world_width if world_width > 0 else 0.5
        norm_y = (world_y - world_min_y) / world_height if world_height > 0 else 0.5
        
        # Scale to SVG coordinates
        svg_x = padding + norm_x * svg_width
        svg_y = padding + norm_y * svg_height
        
        logger.info(f"🔄 Coordinate transform: world({world_x:.1f}, {world_y:.1f}) -> SVG({svg_x:.1f}, {svg_y:.1f})")
        return (svg_x, svg_y)

    def transform_all_route_points(self, route_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Transform all route points to SVG coordinates"""
        return [self.transform_coordinates_to_svg(point) for point in route_points]


    ##############
    def calculate_collision_point(self, route_json: Dict) -> Tuple[float, float]:
        """Calculate the collision point coordinates in world space, then transform to SVG"""
        
        # Get collision point in world coordinates (existing logic)
        if 'route_json' in route_json and isinstance(route_json['route_json'], dict):
            working_data = route_json['route_json']
        else:
            working_data = route_json
        
        collision_coords = working_data.get('collision', {}).get('point', {}).get('coordinates')
        
        if collision_coords and isinstance(collision_coords, dict):
            try:
                world_x = float(collision_coords.get('x', 0))
                world_y = float(collision_coords.get('y', 0))
                world_collision_point = (world_x, world_y)
            except (ValueError, TypeError):
                world_collision_point = None
        else:
            world_collision_point = None
        
        # Fallback: calculate intersection center from lane data
        if not world_collision_point and self.lane_tree_data:
            all_x_coords = []
            all_y_coords = []
            
            for lane in self.lane_tree_data['lane_trees']:
                points = lane.get('points', [])
                if not points:
                    continue
                
                for point in points:
                    try:
                        if isinstance(point, (tuple, list)) and len(point) >= 2:
                            x, y = float(point[0]), float(point[1])
                            all_x_coords.append(x)
                            all_y_coords.append(y)
                        elif isinstance(point, dict) and 'x' in point and 'y' in point:
                            x, y = float(point['x']), float(point['y'])
                            all_x_coords.append(x)
                            all_y_coords.append(y)
                    except (ValueError, TypeError, IndexError):
                        continue
            
            if all_x_coords and all_y_coords:
                world_x = sum(all_x_coords) / len(all_x_coords)
                world_y = sum(all_y_coords) / len(all_y_coords)
                world_collision_point = (world_x, world_y)
                logger.info(f"📍 Calculated world collision point: ({world_x:.1f}, {world_y:.1f})")
        
        # Transform to SVG coordinates
        if world_collision_point:
            svg_collision_point = self.transform_coordinates_to_svg(world_collision_point)
            logger.info(f"📍 SVG collision point: ({svg_collision_point[0]:.1f}, {svg_collision_point[1]:.1f})")
            return svg_collision_point
        
        # Ultimate fallback: return center of SVG canvas
        default_point = (self.svg_width / 2, self.svg_height / 2)
        logger.info(f"📍 Using default SVG collision point: {default_point}")
        return default_point    



    def create_animated_svg(self, scenario: CollisionScenario) -> str:
        """Create the animated SVG content"""
        
        dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
        
        # Add background map if available
        self.add_background_map(dwg, scenario.base_map_image)
        
        # Add lane markings
        self.add_lane_markings(dwg)
        
        # Add vehicle animations
        self.add_vehicle_animation(dwg, scenario.user_vehicle, "blue", "User Vehicle")
        self.add_vehicle_animation(dwg, scenario.other_vehicle, "red", "Other Vehicle")
        
        # Add collision point marker
        self.add_collision_marker(dwg, scenario.collision_point)
        
        # Add timeline and controls
        self.add_animation_controls(dwg)
        
        return dwg.tostring()

    def add_background_map(self, dwg, map_image: str):
        """Add background map image"""
        try:
            # Load map image from S3 and embed as base64
            map_key = f"outputs/{self.connection_id}/{map_image}"
            response = self.s3.get_object(Bucket=self.bucket, Key=map_key)
            
            # For now, just add a placeholder rectangle
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(self.svg_width, self.svg_height),
                fill='lightgray',
                opacity=0.3
            ))
            
        except Exception as e:
            logger.warning(f"Could not load background map: {e}")
            # Add fallback background
            dwg.add(dwg.rect(
                insert=(0, 0),
                size=(self.svg_width, self.svg_height),
                fill='lightgray'
            ))

    def add_lane_markings(self, dwg):
        """Add lane markings based on lane tree data"""
        if not self.lane_tree_data:
            return
        
        for lane in self.lane_tree_data['lane_trees']:
            points = lane.get('points', [])
            if len(points) >= 2:
                # 🔥 FIX: Transform points to SVG coordinates
                svg_points = []
                for point in points:
                    try:
                        if isinstance(point, (tuple, list)) and len(point) >= 2:
                            world_point = (float(point[0]), float(point[1]))
                            svg_point = self.transform_coordinates_to_svg(world_point)
                            svg_points.append(svg_point)
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if len(svg_points) >= 2:
                    # Convert points to SVG path
                    path_data = f"M{svg_points[0][0]},{svg_points[0][1]}"
                    for point in svg_points[1:]:
                        path_data += f" L{point[0]},{point[1]}"
                    
                    dwg.add(dwg.path(
                        d=path_data,
                        stroke='white',
                        stroke_width=2,
                        fill='none',
                        stroke_dasharray="5,5"
                    ))


    def add_vehicle_animation(self, dwg, vehicle_path: VehiclePath, color: str, label: str):
        """Add animated vehicle following the calculated path"""
        
        if not vehicle_path.route_points:
            return
        
        # Create vehicle shape (simple rectangle/circle)
        vehicle = dwg.circle(
            center=(vehicle_path.route_points[0][0], vehicle_path.route_points[0][1]),
            r=8,
            fill=color,
            stroke='black',
            stroke_width=2
        )
        
        # Add vehicle label
        label_text = dwg.text(
            label,
            insert=(vehicle_path.route_points[0][0] - 20, vehicle_path.route_points[0][1] - 15),
            fill='black',
            font_size=12
        )
        
        # Create animation path
        if len(vehicle_path.route_points) > 1:
            path_data = f"M{vehicle_path.route_points[0][0]},{vehicle_path.route_points[0][1]}"
            for point in vehicle_path.route_points[1:]:
                path_data += f" L{point[0]},{point[1]}"
            
            # Add path visualization
            path_visual = dwg.path(
                d=path_data,
                stroke=color,
                stroke_width=3,
                fill='none',
                opacity=0.5
            )
            dwg.add(path_visual)
            
            # Add animation
            animate_motion = dwg.animateMotion(
                path=path_data,
                dur="8s",
                repeatCount="indefinite"
            )
            vehicle.add(animate_motion)
            
            # Animate the label too
            label_animate = dwg.animateMotion(
                path=path_data,
                dur="8s", 
                repeatCount="indefinite"
            )
            label_text.add(label_animate)
        
        dwg.add(vehicle)
        dwg.add(label_text)

    def add_collision_marker(self, dwg, collision_point: Tuple[float, float]):
        """Add collision point marker with animation"""
        
        # Collision explosion effect
        explosion = dwg.circle(
            center=collision_point,
            r=5,
            fill='orange',
            opacity=0
        )
        
        # Animation: appears at collision time
        explosion.add(dwg.animate(
            attributeName='r',
            values='5;30;5',
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
        
        # Permanent collision marker
        marker = dwg.text(
            "⚠️ COLLISION",
            insert=(collision_point[0] - 30, collision_point[1] + 40),
            fill='red',
            font_size=14,
            font_weight='bold'
        )
        dwg.add(marker)

    def add_animation_controls(self, dwg):
        """Add animation timeline and controls"""
        
        # Timeline background
        timeline_bg = dwg.rect(
            insert=(50, self.svg_height - 80),
            size=(self.svg_width - 100, 60),
            fill='black',
            opacity=0.7,
            rx=10
        )
        dwg.add(timeline_bg)
        
        # Timeline labels
        labels = [
            "0s: Vehicles approach",
            "3s: Enter intersection", 
            "6s: COLLISION!",
            "8s: Animation repeats"
        ]
        
        for i, label in enumerate(labels):
            x_pos = 70 + i * (self.svg_width - 140) / len(labels)
            dwg.add(dwg.text(
                label,
                insert=(x_pos, self.svg_height - 50),
                fill='white',
                font_size=10
            ))

    def calculate_end_direction(self, start_direction: str, maneuver: str) -> str:
        """Calculate end direction based on start direction and maneuver"""
        
        direction_map = {
            "north": {"left_turn": "west", "right_turn": "east", "straight": "north"},
            "south": {"left_turn": "east", "right_turn": "west", "straight": "south"},
            "east": {"left_turn": "north", "right_turn": "south", "straight": "east"},
            "west": {"left_turn": "south", "right_turn": "north", "straight": "west"}
        }
        
        return direction_map.get(start_direction, {}).get(maneuver, start_direction)

    ##############
    # JSON Parsing with Flexible Structure Handling (NEW)
    ##############
    def parse_route_scenario(self, route_json: Dict) -> Dict:
        """Parse and validate the route scenario JSON with nested structure support"""
        
        # 🔥 FIX: Handle nested route_json structure
        working_data = route_json
        if 'route_json' in route_json and isinstance(route_json['route_json'], dict):
            working_data = route_json['route_json']
            logger.info("✅ Found nested route_json structure, using nested data")
        
        # Try to find vehicles data in the working data
        vehicles_data = working_data.get('vehicles', {})
        
        # If no 'vehicles' key, try alternative structures
        if not vehicles_data:
            logger.warning("⚠️ No 'vehicles' key found, trying alternative structures...")
            
            # Try alternative structures in working_data first
            user_vehicle = (working_data.get('user_vehicle') or 
                        working_data.get('vehicle1') or 
                        working_data.get('primary_vehicle') or
                        working_data.get('car1'))
                        
            other_vehicle = (working_data.get('other_vehicle') or 
                            working_data.get('vehicle2') or 
                            working_data.get('secondary_vehicle') or
                            working_data.get('car2'))
            
            # If still not found, try the original route_json level
            if not user_vehicle:
                user_vehicle = (route_json.get('user_vehicle') or 
                            route_json.get('vehicle1') or 
                            route_json.get('primary_vehicle') or
                            route_json.get('car1'))
                            
            if not other_vehicle:
                other_vehicle = (route_json.get('other_vehicle') or 
                                route_json.get('vehicle2') or 
                                route_json.get('secondary_vehicle') or
                                route_json.get('car2'))
            
            # 🔥 NEW: Try to extract from user_path if vehicles still not found
            if not user_vehicle and 'user_path' in route_json:
                user_path = route_json['user_path']
                other_vehicle_info = user_path.get('other_vehicle', {})
                
                # Create user vehicle from user_path
                user_vehicle = {
                    'path': {
                        'origin': {
                            'direction': user_path.get('origin_direction', 'unknown'),
                            'road_name': user_path.get('origin_road', 'unknown')[:100] if user_path.get('origin_road') else 'unknown',
                            'lane': user_path.get('approach_lane', 'unknown')
                        },
                        'intended_destination': {
                            'maneuver': user_path.get('intended_maneuver', 'straight'),
                            'target_road': user_path.get('destination_road', 'unknown')
                        }
                    },
                    'state_at_collision': {
                        'speed': 'unknown',
                        'position': user_path.get('collision_point', 'intersection'),
                        'action': user_path.get('intended_maneuver', 'straight')
                    }
                }
                
                # Create other vehicle from other_vehicle data in user_path
                other_vehicle = {
                    'path': {
                        'origin': {
                            'direction': 'unknown',
                            'road_name': 'unknown',
                            'lane': 'unknown'
                        },
                        'intended_destination': {
                            'maneuver': other_vehicle_info.get('action', 'unknown'),
                            'target_road': 'unknown'
                        }
                    },
                    'state_at_collision': {
                        'speed': 'unknown',
                        'position': other_vehicle_info.get('position', 'unknown'),
                        'action': other_vehicle_info.get('action', 'unknown')
                    }
                }
                
                logger.info("🔧 Created vehicles from user_path structure")
            
            vehicles_data = {
                'user_vehicle': user_vehicle or {},
                'other_vehicle': other_vehicle or {}
            }
            
            logger.info(f"🔄 Using alternative structure: user={bool(user_vehicle)}, other={bool(other_vehicle)}")
        else:
            logger.info("✅ Found vehicles data in standard location")
        
        # Handle collision data from working_data
        collision_data = working_data.get('collision', {})
        if not collision_data:
            # Try alternative collision keys
            collision_data = (working_data.get('accident', {}) or 
                            working_data.get('crash', {}) or 
                            working_data.get('impact', {}) or
                            route_json.get('collision', {}) or
                            route_json.get('accident', {}) or
                            route_json.get('crash', {}))
        
        # Handle timeline data
        timeline_data = working_data.get('timeline', {})
        if not timeline_data:
            timeline_data = route_json.get('timeline', {})
        
        result = {
            "user_vehicle": vehicles_data.get('user_vehicle', {}),
            "other_vehicle": vehicles_data.get('other_vehicle', {}),
            "collision": collision_data,
            "timeline": timeline_data
        }
        
        # 🔥 DEBUG: Log what we extracted
        logger.info(f"🔍 Extracted vehicles: user={bool(result['user_vehicle'])}, other={bool(result['other_vehicle'])}")
        if result['user_vehicle']:
            user_path = result['user_vehicle'].get('path', {})
            logger.info(f"👤 User vehicle: {user_path.get('origin', {}).get('direction', 'unknown')} -> {user_path.get('intended_destination', {}).get('maneuver', 'unknown')}")
        if result['other_vehicle']:
            other_path = result['other_vehicle'].get('path', {})
            logger.info(f"🚗 Other vehicle: {other_path.get('origin', {}).get('direction', 'unknown')} -> {other_path.get('intended_destination', {}).get('maneuver', 'unknown')}")
        
        return result


def lambda_handler(event, context):
    """S3-only entry (kept only if you still invoke this as a Lambda)."""
    try:
        bucket_name   = os.getenv("BUCKET_NAME") or event.get("bucket_name")
        connection_id = os.getenv("CONNECTION_ID") or event.get("connection_id")
        route_s3_key  = os.getenv("ROUTE_S3_KEY") or event.get("route_s3_key")
        lane_tree_key = os.getenv("LANE_TREE_KEY")  # optional

        if not route_s3_key and connection_id:
            route_s3_key = f"animation-data/{connection_id}/complete_route_data.json"

        if not (bucket_name and connection_id and route_s3_key):
            raise RuntimeError("Missing BUCKET_NAME / CONNECTION_ID / ROUTE_S3_KEY")

        logger.info(f"📥 Loading route_json from s3://{bucket_name}/{route_s3_key}")
        route_json = _load_json_from_s3(bucket_name, route_s3_key)

        gen = SVGAnimationGenerator(bucket_name, connection_id)
        if lane_tree_key:
            gen.lane_tree_key_override = lane_tree_key

        svg_key = gen.generate_accident_animation(route_json)
        return {"statusCode": 200, "svg_key": svg_key, "message": "SVG animation generated successfully"}
    except Exception as e:
        logger.error(f"❗ SVG generation failed: {e}")
        return {"statusCode": 500, "error": str(e)}


if __name__ == "__main__":
    try:
        # Prefer env (since ECS passes env). Accept argv as a fallback for local runs.
        if len(sys.argv) >= 3:
            connection_id = sys.argv[1]
            bucket_name   = sys.argv[2]
            route_s3_key  = os.getenv("ROUTE_S3_KEY") or \
                            (sys.argv[3] if len(sys.argv) >= 4 else f"animation-data/{connection_id}/complete_route_data.json")
        else:
            bucket_name   = _get_env("BUCKET_NAME")
            connection_id = _get_env("CONNECTION_ID")
            route_s3_key  = _get_env("ROUTE_S3_KEY", required=False) \
                            or f"animation-data/{connection_id}/complete_route_data.json"

        lane_tree_key = os.getenv("LANE_TREE_KEY")

        logger.info(f"🚀 Starting SVG generation: conn={connection_id}")
        logger.info(f"📥 route_json: s3://{bucket_name}/{route_s3_key}")
        if lane_tree_key:
            logger.info(f"📥 lane_tree:  s3://{bucket_name}/{lane_tree_key}")

        route_json = _load_json_from_s3(bucket_name, route_s3_key)

        gen = SVGAnimationGenerator(bucket_name, connection_id)
        if lane_tree_key:
            gen.lane_tree_key_override = lane_tree_key

        out_key = gen.generate_accident_animation(route_json)
        print(json.dumps({"statusCode": 200, "svg_key": out_key}, indent=2))
    except Exception as e:
        logger.error(f"❗ SVG generation failed: {e}")
        print(json.dumps({"statusCode": 500, "error": str(e)}, indent=2))
        raise