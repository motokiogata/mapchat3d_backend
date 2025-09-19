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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
        
    def generate_accident_animation(self, route_json: Dict) -> str:
        """Main method to generate animated SVG"""
        try:
            # 1. Load lane tree data
            self.load_lane_tree_data()
            
            # 2. Parse route scenario
            scenario = self.parse_route_scenario(route_json)
            
            # 3. Match vehicles to lane paths
            user_path = self.match_vehicle_to_lanes(
                route_json['vehicles']['user_vehicle'], 'user'
            )
            other_path = self.match_vehicle_to_lanes(
                route_json['vehicles']['other_vehicle'], 'other'
            )
            
            # 4. Generate collision scenario
            scenario = CollisionScenario(
                user_vehicle=user_path,
                other_vehicle=other_path,
                collision_point=self.calculate_collision_point(route_json),
                collision_type=route_json['collision']['type'],
                base_map_image=f"{self.connection_id}_intersection_map.png"
            )
            
            # 5. Create animated SVG
            svg_content = self.create_animated_svg(scenario)
            
            # 6. Save to S3
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
        """Load the lane tree routes enhanced JSON file"""
        try:
            lane_tree_key = f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"
            
            response = self.s3.get_object(Bucket=self.bucket, Key=lane_tree_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"✅ Loaded lane tree data with {len(self.lane_tree_data['lane_trees'])} lanes")
            
        except Exception as e:
            logger.error(f"❗ Failed to load lane tree data: {e}")
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
            # Use actual map coordinates
            start_points = [(p['x'], p['y']) for p in map_coords[:len(start_points)//2]]
        else:
            # Use lane tree points directly
            start_points = start_points[:len(start_points)//2]  # Take first half for approach
        
        route_points.extend(start_points)
        
        # Add intersection transition points
        if end_lane and maneuver != "straight":
            # Generate turning curve
            turn_points = self.generate_turn_curve(
                start_points[-1] if start_points else (0, 0),
                end_lane.get('points', [])[:3] if end_lane.get('points') else [(100, 100)],
                maneuver
            )
            route_points.extend(turn_points)
        
        # Smooth the route
        smoothed_points = self.smooth_route_points(route_points)
        
        return smoothed_points

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

    def calculate_collision_point(self, route_json: Dict) -> Tuple[float, float]:
        """Calculate the collision point coordinates"""
        
        # Try to get from route JSON
        collision_coords = route_json.get('collision', {}).get('point', {}).get('coordinates')
        
        if collision_coords and isinstance(collision_coords, dict):
            return (collision_coords.get('x', 0), collision_coords.get('y', 0))
        
        # Fallback: calculate intersection center
        if self.lane_tree_data:
            all_points = []
            for lane in self.lane_tree_data['lane_trees']:
                if lane.get('points'):
                    all_points.extend(lane['points'])
            
            if all_points:
                center_x = sum(p[0] if isinstance(p, tuple) else p for p in all_points) / len(all_points)
                center_y = sum(p[1] if isinstance(p, tuple) else p for p in all_points) / len(all_points)
                return (center_x, center_y)
        
        return (400, 300)  # Default center

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
                # Convert points to SVG path
                path_data = f"M{points[0][0]},{points[0][1]}"
                for point in points[1:]:
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

    def parse_route_scenario(self, route_json: Dict) -> Dict:
        """Parse and validate the route scenario JSON"""
        return {
            "user_vehicle": route_json.get('vehicles', {}).get('user_vehicle', {}),
            "other_vehicle": route_json.get('vehicles', {}).get('other_vehicle', {}),
            "collision": route_json.get('collision', {}),
            "timeline": route_json.get('timeline', {})
        }


def lambda_handler(event, context):
    """Lambda handler for triggering SVG generation"""
    try:
        connection_id = event['connection_id']
        bucket_name = event['bucket_name']
        
        # Get route JSON from DynamoDB
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table("AccidentDataTable")
        
        response = table.get_item(Key={"connection_id": connection_id})
        if 'Item' not in response:
            raise Exception("Route data not found")
        
        route_json = response['Item']['route_analysis']['route_json']
        
        # Generate SVG animation
        generator = SVGAnimationGenerator(bucket_name, connection_id)
        svg_key = generator.generate_accident_animation(route_json)
        
        return {
            "statusCode": 200,
            "svg_key": svg_key,
            "message": "SVG animation generated successfully"
        }
        
    except Exception as e:
        logger.error(f"❗ SVG generation failed: {e}")
        return {
            "statusCode": 500,
            "error": str(e)
        }


if __name__ == "__main__":
    # For Docker container execution
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python svg_generator.py <connection_id> <bucket_name>")
        sys.exit(1)
    
    connection_id = sys.argv[1]
    bucket_name = sys.argv[2]
    
    event = {
        "connection_id": connection_id,
        "bucket_name": bucket_name
    }
    
    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2))