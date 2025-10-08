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

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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

# 🔧 SIMPLIFIED: EnhancedSVGAnimationGenerator with clean implementation
class EnhancedSVGAnimationGenerator:

    def __init__(self, s3_bucket: str, connection_id: str):
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client("bedrock-runtime", 
                                   region_name=os.environ.get("AWS_REGION", "us-east-1"))
        self.bucket = s3_bucket
        self.connection_id = connection_id
        self.svg_width = 1280
        self.svg_height = 1280
        
        # Lane tree data storage
        self.lane_tree_data = None
        self.use_llm_validation = True

    # 🚀 SIMPLIFIED: Only try S3 for lane tree data
    def load_lane_tree_data_from_s3(self):
        """Load lane tree data from S3 only - with detailed logging"""
        logger.info("🌐 Loading lane tree data from S3...")
        
        try:
            s3_key = f"outputs/{self.connection_id}/{self.connection_id}_lane_tree_routes_enhanced.json"
            logger.info(f"📍 S3 Key: {s3_key}")
            
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            self.lane_tree_data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Detailed logging
            if self.lane_tree_data:
                lane_trees = self.lane_tree_data.get('lane_trees', [])
                logger.info(f"✅ S3 lane tree loaded successfully: {len(lane_trees)} lanes")
                
                # Log all available lane IDs for debugging
                lane_ids = [lane.get('lane_id', 'NO_ID') for lane in lane_trees]
                logger.info(f"📋 Available lane IDs: {lane_ids}")
                
                # Log sample lane data
                if lane_trees:
                    sample_lane = lane_trees[0]
                    sample_points = sample_lane.get('points', [])
                    logger.info(f"🔍 Sample lane '{sample_lane.get('lane_id')}': {len(sample_points)} points")
                    if sample_points:
                        logger.info(f"🔍 First point example: {sample_points[0]}")
            else:
                logger.warning("⚠️ Lane tree data loaded but is empty")
                
        except Exception as e:
            logger.error(f"❗ S3 lane tree loading failed: {e}")
            logger.error(f"❗ Exception type: {type(e).__name__}")
            self.lane_tree_data = None

    # 🚀 SIMPLIFIED: Direct exact match only (no fuzzy matching)
    def extract_waypoints_for_lane_exact(self, lane_id: str) -> list:
        """Extract waypoints with exact lane_id match only - with detailed logging"""
        logger.info(f"🎯 Extracting waypoints for exact lane_id: '{lane_id}'")
        
        if not lane_id or lane_id in ['unknown', 'None', '', None]:
            logger.warning(f"⚠️ Invalid lane_id provided: '{lane_id}'")
            return []
        
        if not self.lane_tree_data:
            logger.error("❗ No lane tree data available")
            return []
        
        lane_trees = self.lane_tree_data.get('lane_trees', [])
        if not lane_trees:
            logger.error("❗ No lane_trees found in data")
            return []
        
        logger.info(f"🔍 Searching {len(lane_trees)} lanes for exact match...")
        
        # Exact match search
        for i, lane in enumerate(lane_trees):
            current_lane_id = lane.get('lane_id', '')
            logger.info(f"🔍 Lane {i+1}/{len(lane_trees)}: '{current_lane_id}' vs target '{lane_id}'")
            
            if current_lane_id == lane_id:
                logger.info(f"✅ EXACT MATCH FOUND: '{current_lane_id}'")
                
                points = lane.get('points', [])
                logger.info(f"📊 Lane has {len(points)} raw points")
                
                if not points:
                    logger.warning(f"⚠️ Lane '{lane_id}' found but has no points")
                    return []
                
                # Convert points to waypoints
                waypoints = []
                for j, point in enumerate(points):
                    try:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            waypoint = {'x': float(point[0]), 'y': float(point[1])}
                        elif isinstance(point, dict) and 'x' in point and 'y' in point:
                            waypoint = {'x': float(point['x']), 'y': float(point['y'])}
                        else:
                            logger.warning(f"⚠️ Invalid point format at index {j}: {point} (type: {type(point)})")
                            continue
                        
                        waypoints.append(waypoint)
                        
                        # Log first few waypoints for debugging
                        if j < 3:
                            logger.info(f"🔍 Waypoint {j}: {waypoint}")
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Could not convert point {j}: {e}")
                        continue
                
                logger.info(f"✅ Successfully extracted {len(waypoints)} valid waypoints")
                return waypoints
        
        # No match found
        logger.error(f"❌ NO EXACT MATCH found for lane_id: '{lane_id}'")
        available_lanes = [lane.get('lane_id', 'NO_ID') for lane in lane_trees]
        logger.error(f"📋 Available lanes were: {available_lanes}")
        
        return []

    # 🚀 CORE: Main animation method with detailed logging
    def generate_analytics_guided_animation_clean(self) -> str:
        """CLEAN: Analytics-guided animation with detailed logging"""
        logger.info("🎯 Starting CLEAN analytics-guided animation")
        
        try:
            # Step 1: Load LLM data
            logger.info("📂 STEP 1: Loading LLM ready data...")
            llm_data = self.load_llm_ready_data()
            
            if not llm_data:
                logger.error("❗ No LLM data available, creating fallback")
                return self.create_dual_fallback_animation()
            
            logger.info(f"✅ LLM data loaded with keys: {list(llm_data.keys())}")
            
            # Step 2: Load lane tree data from S3
            logger.info("📂 STEP 2: Loading lane tree data from S3...")
            self.load_lane_tree_data_from_s3()
            
            if not self.lane_tree_data:
                logger.error("❗ No lane tree data available, creating fallback")
                return self.create_dual_fallback_animation()
            
            # Step 3: Process user vehicle
            logger.info("🚙 STEP 3: Processing user vehicle...")
            user_path_data = llm_data.get('user_path', {})
            
            if not user_path_data:
                logger.error("❗ No user_path in LLM data")
                return self.create_dual_fallback_animation()
            
            user_approach_lane = user_path_data.get('approach_lane', '')
            user_lane_id = user_path_data.get('lane_id', user_approach_lane)  # Try both fields
            user_maneuver = user_path_data.get('intended_maneuver', 'unknown')
            
            logger.info(f"🎯 User vehicle details:")
            logger.info(f"   - approach_lane: '{user_approach_lane}'")
            logger.info(f"   - lane_id: '{user_lane_id}'")
            logger.info(f"   - intended_maneuver: '{user_maneuver}'")
            
            # Extract user waypoints
            target_user_lane = user_lane_id or user_approach_lane
            user_waypoints = self.extract_waypoints_for_lane_exact(target_user_lane)
            
            # 🚀 NEVER DROP BLUE CAR
            if not user_waypoints:
                logger.warning(f"⚠️ No waypoints found for user lane '{target_user_lane}', creating fallback marker")
                user_waypoints = self.create_fallback_waypoints('user')
                user_label = f"Your Vehicle (fallback marker - {user_maneuver})"
                logger.info("🔵 Fallback marker created for user vehicle")
            else:
                user_label = f"Your Vehicle ({user_maneuver})"
                logger.info(f"🔵 User vehicle: {len(user_waypoints)} real waypoints")
            
            # Step 4: Process other vehicle
            logger.info("🚛 STEP 4: Processing other vehicle...")
            other_path_data = llm_data.get('other_vehicle_path', {})
            
            if not other_path_data:
                logger.warning("⚠️ No other_vehicle_path in LLM data, creating fallback")
                other_waypoints = self.create_fallback_waypoints('other')
                other_label = "Other Vehicle (no data - fallback marker)"
            else:
                other_approach_lane = other_path_data.get('approach_lane', '')
                other_lane_id = other_path_data.get('lane_id', other_approach_lane)
                other_maneuver = other_path_data.get('intended_maneuver', 'unknown')
                
                logger.info(f"🎯 Other vehicle details:")
                logger.info(f"   - approach_lane: '{other_approach_lane}'")
                logger.info(f"   - lane_id: '{other_lane_id}'")
                logger.info(f"   - intended_maneuver: '{other_maneuver}'")
                
                target_other_lane = other_lane_id or other_approach_lane
                other_waypoints = self.extract_waypoints_for_lane_exact(target_other_lane)
                
                if not other_waypoints:
                    logger.warning(f"⚠️ No waypoints found for other lane '{target_other_lane}', creating fallback marker")
                    other_waypoints = self.create_fallback_waypoints('other')
                    other_label = f"Other Vehicle (fallback marker - {other_maneuver})"
                    logger.info("🔴 Fallback marker created for other vehicle")
                else:
                    other_label = f"Other Vehicle ({other_maneuver})"
                    logger.info(f"🔴 Other vehicle: {len(other_waypoints)} real waypoints")
            
            # Step 5: Create vehicle path objects
            logger.info("🛣️ STEP 5: Creating vehicle path objects...")
            
            user_path = {
                'waypoints': self.convert_waypoints_to_svg(user_waypoints),
                'raw_waypoints': user_waypoints,
                'color': 'blue',
                'label': user_label,
                'vehicle_type': 'user'
            }
            
            other_path = {
                'waypoints': self.convert_waypoints_to_svg(other_waypoints),
                'raw_waypoints': other_waypoints,
                'color': 'red',
                'label': other_label,
                'vehicle_type': 'other'
            }
            
            logger.info(f"✅ User path: {len(user_path['waypoints'])} SVG waypoints")
            logger.info(f"✅ Other path: {len(other_path['waypoints'])} SVG waypoints")
            
            # Step 6: Calculate collision and create animation
            logger.info("💥 STEP 6: Creating collision animation...")
            collision_data = self.calculate_collision_timing(user_path, other_path)
            svg_content = self.create_dual_vehicle_svg_animation(user_path, other_path, collision_data)
            
            # Step 7: Save to S3
            logger.info("💾 STEP 7: Saving animation to S3...")
            svg_key = f"animations/{self.connection_id}/clean_analytics_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=svg_content,
                ContentType='image/svg+xml'
            )
            
            # Final success logging
            logger.info("🎉 ANIMATION CREATION SUCCESS!")
            logger.info(f"📁 SVG saved to: {svg_key}")
            logger.info(f"🔵 Blue car: {user_label}")
            logger.info(f"🔴 Red car: {other_label}")
            logger.info(f"📊 Total lanes loaded: {len(self.lane_tree_data.get('lane_trees', []))}")
            
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ CLEAN animation failed: {e}")
            import traceback
            logger.error(f"❗ Full traceback: {traceback.format_exc()}")
            return self.create_dual_fallback_animation()

    # 🚀 UPDATED: Main entry point
    def generate_intelligent_animation(self) -> str:
        """Main method - use clean implementation"""
        try:
            mode = os.environ.get("MODE", "CLEAN_ANALYTICS")
            logger.info(f"🎯 Using mode: {mode}")
            
            return self.generate_analytics_guided_animation_clean()
                
        except Exception as e:
            logger.error(f"❗ Animation generation failed: {e}")
            return self.create_dual_fallback_animation()

    # 🚀 HELPER METHODS (simplified)
    
    def load_llm_ready_data(self) -> dict:
        """Load the lightweight metadata-only data"""
        try:
            llm_key = f"animation-data/{self.connection_id}/llm_ready_data.json"
            logger.info(f"📍 Loading LLM data from: {llm_key}")
            
            response = self.s3.get_object(Bucket=self.bucket, Key=llm_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"✅ LLM data loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"❗ Failed to load LLM data: {e}")
            return {}

    def create_fallback_waypoints(self, vehicle_type: str) -> list:
        """Create fallback waypoints when lane extraction fails"""
        if vehicle_type == 'user':
            waypoints = [
                {'x': 100, 'y': self.svg_height // 2},
                {'x': self.svg_width // 2 - 50, 'y': self.svg_height // 2},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2}
            ]
        else:  # other vehicle
            waypoints = [
                {'x': self.svg_width // 2, 'y': 100},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2 - 50},
                {'x': self.svg_width // 2, 'y': self.svg_height // 2}
            ]
        
        logger.info(f"🎯 Created {len(waypoints)} fallback waypoints for {vehicle_type} vehicle")
        return waypoints

    # 🚀 UPDATED: Direct 1:1 mapping with bounds checking
    def convert_waypoints_to_svg(self, waypoints: list) -> list:
        """Direct 1:1 coordinate mapping - no scaling needed"""
        logger.info(f"🎯 1:1 Mapping: {len(waypoints)} waypoints (1280x1280)")
        
        if not waypoints:
            return [{'x': 640, 'y': 640}]  # Center of 1280x1280
        
        svg_waypoints = []
        for i, point in enumerate(waypoints):
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                # Direct mapping with bounds checking
                svg_x = max(0, min(1280, point['x']))
                svg_y = max(0, min(1280, point['y']))
                
                svg_waypoints.append({'x': svg_x, 'y': svg_y})
                
                if i < 3:
                    logger.info(f"🎯 Point {i}: ({point['x']}, {point['y']}) → ({svg_x}, {svg_y})")
        
        logger.info(f"✅ 1:1 mapping complete: {len(svg_waypoints)} points")
        return svg_waypoints


    def create_dual_fallback_animation(self) -> str:
        """Create dual-vehicle fallback animation when everything fails"""
        logger.info("🆘 Creating dual fallback animation")
        
        try:
            # Create fallback paths
            user_path = {
                'waypoints': [
                    {'x': 100, 'y': self.svg_height // 2},
                    {'x': self.svg_width // 2 - 50, 'y': self.svg_height // 2},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ],
                'color': 'blue',
                'label': 'Your Vehicle (complete fallback)',
                'vehicle_type': 'user'
            }
            
            other_path = {
                'waypoints': [
                    {'x': self.svg_width // 2, 'y': 100},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2 - 50},
                    {'x': self.svg_width // 2, 'y': self.svg_height // 2}
                ],
                'color': 'red',
                'label': 'Other Vehicle (complete fallback)',
                'vehicle_type': 'other'
            }
            
            collision_data = {
                'collision_point': [self.svg_width // 2, self.svg_height // 2],
                'collision_timing': 5.0
            }
            
            svg_content = self.create_dual_vehicle_svg_animation(user_path, other_path, collision_data)
            
            svg_key = f"animations/{self.connection_id}/dual_fallback_animation.svg"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=svg_key,
                Body=svg_content,
                ContentType='image/svg+xml'
            )
            
            logger.info(f"✅ Dual fallback animation created: {svg_key}")
            return svg_key
            
        except Exception as e:
            logger.error(f"❗ Even fallback failed: {e}")
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

    ##newest one##
    def create_dual_vehicle_svg_animation(self, user_path: dict, other_path: dict, collision_data: dict) -> str:
        """Create SVG with separated <defs> paths - NO PLACEHOLDERS"""
        logger.info("🎬 Creating dual-vehicle SVG animation")
        
        dwg = svgwrite.Drawing(size=(self.svg_width, self.svg_height))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(self.svg_width, self.svg_height), fill='#1a1a1a'))
        self.add_intersection_background(dwg)
        
        # Get waypoints
        user_waypoints = user_path.get('waypoints', [])
        other_waypoints = other_path.get('waypoints', [])
        
        # Create <defs> with paths
        defs = dwg.defs
        user_path_d = self.waypoints_to_svg_path(user_waypoints)
        other_path_d = self.waypoints_to_svg_path(other_waypoints)
        
        defs.add(dwg.path(id="userPath", d=user_path_d))
        defs.add(dwg.path(id="otherPath", d=other_path_d))
        
        # 🚀 FIX: Create vehicles with proper SMIL - NO PLACEHOLDERS
        collision_timing = collision_data.get('collision_timing', 5.0)
        
        # User vehicle - use raw SVG string creation
        user_vehicle_svg = f'''
        <circle cx="0" cy="0" r="12" fill="blue" stroke="white" stroke-width="2">
            <animateMotion dur="{collision_timing}s" repeatCount="1" fill="freeze" begin="0s">
                <mpath href="#userPath"/>
            </animateMotion>
        </circle>'''
        
        # Other vehicle - use raw SVG string creation  
        other_vehicle_svg = f'''
        <circle cx="0" cy="0" r="10" fill="red" stroke="white" stroke-width="2">
            <animateMotion dur="{collision_timing}s" repeatCount="1" fill="freeze" begin="0s">
                <mpath href="#otherPath"/>
            </animateMotion>
        </circle>'''
        
        # Add collision effect, labels, timeline
        self.add_collision_effect(dwg, collision_data)
        self.add_dual_vehicle_labels(dwg, user_path, other_path)
        self.add_timeline_display(dwg)
        
        # 🚀 FIX: Inject raw SVG strings into the final output
        svg_string = dwg.tostring()
        
        # Insert vehicles before closing </svg> tag
        vehicles_svg = user_vehicle_svg + other_vehicle_svg
        svg_string = svg_string.replace('</svg>', vehicles_svg + '</svg>')
        
        logger.info("✅ SVG created with separated <defs> paths and proper mpath")
        return svg_string


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

    def determine_path_direction(self, waypoints: list, collision_point: list) -> tuple:
        """
        Determine if waypoints are in correct direction toward collision point
        Returns: (corrected_waypoints, direction_was_reversed)
        """
        if not waypoints or len(waypoints) < 2:
            return waypoints, False
        
        # Calculate distance from first point to collision
        first_point = waypoints[0]
        if isinstance(first_point, dict):
            first_coords = [first_point['x'], first_point['y']]
        else:
            first_coords = first_point
        
        first_to_collision = math.sqrt(
            (first_coords[0] - collision_point[0])**2 + 
            (first_coords[1] - collision_point[1])**2
        )
        
        # Calculate distance from last point to collision  
        last_point = waypoints[-1]
        if isinstance(last_point, dict):
            last_coords = [last_point['x'], last_point['y']]
        else:
            last_coords = last_point
            
        last_to_collision = math.sqrt(
            (last_coords[0] - collision_point[0])**2 + 
            (last_coords[1] - collision_point[1])**2
        )
        
        logger.info(f"🎯 Direction check: First→Collision={first_to_collision:.1f}, Last→Collision={last_to_collision:.1f}")
        
        # If last point is closer to collision, waypoints are in correct direction
        # If first point is closer to collision, waypoints are reversed
        if first_to_collision < last_to_collision:
            logger.info("🔄 REVERSED: Waypoints are backwards - fixing direction")
            return list(reversed(waypoints)), True
        else:
            logger.info("✅ CORRECT: Waypoints are in right direction")
            return waypoints, False

    def truncate_path_at_collision(self, waypoints: list, collision_point: list) -> list:
        """Truncate vehicle path to end at collision point - with direction handling"""
        if not waypoints or not collision_point:
            return waypoints
        
        # 🚀 STEP 1: Fix direction first
        corrected_waypoints, was_reversed = self.determine_path_direction(waypoints, collision_point)
        
        # 🚀 STEP 2: Find collision point in corrected path
        closest_index = len(corrected_waypoints) - 1
        min_distance = float('inf')
        
        for i, point in enumerate(corrected_waypoints):
            if isinstance(point, dict):
                point_coords = [point.get('x', 0), point.get('y', 0)]
            else:
                point_coords = point
                
            distance = math.sqrt(
                (point_coords[0] - collision_point[0])**2 + 
                (point_coords[1] - collision_point[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 🚀 STEP 3: Create path from start to collision
        animation_waypoints = corrected_waypoints[:closest_index + 1]
        
        # Add exact collision point as final destination
        animation_waypoints.append({
            'x': collision_point[0], 
            'y': collision_point[1]
        })
        
        logger.info(f"🎯 Path processing: {len(waypoints)} → {len(animation_waypoints)} waypoints")
        logger.info(f"🔄 Direction reversed: {was_reversed}")
        logger.info(f"📍 Collision at index: {closest_index}")
        
        return animation_waypoints
        

    def add_vehicle_animation(self, dwg, vehicle_path: dict, collision_data: dict, vehicle_type: str):
        """Add animated vehicle to SVG - WITH DIRECTION DETECTION"""
        waypoints = vehicle_path.get('waypoints', [])
        if not waypoints:
            return
        
        color = vehicle_path.get('color', 'gray')
        vehicle_size = 12 if vehicle_type == 'user' else 10
        
        # 🚀 FIX: Get collision point and create proper animation path
        collision_point = collision_data.get('collision_point', [self.svg_width//2, self.svg_height//2])
        animation_waypoints = self.truncate_path_at_collision(waypoints, collision_point)
        
        if not animation_waypoints:
            logger.warning(f"⚠️ No animation waypoints for {vehicle_type} vehicle")
            return
        
        # Start vehicle at correct starting position
        start_point = animation_waypoints[0]
        vehicle = dwg.circle(
            center=(start_point['x'] if isinstance(start_point, dict) else start_point[0],
                    start_point['y'] if isinstance(start_point, dict) else start_point[1]),
            r=vehicle_size,
            fill=color,
            stroke='white',
            stroke_width=2
        )
        
        # Create animation path - from start to collision point
        if len(animation_waypoints) > 1:
            animation_path = self.waypoints_to_svg_path(animation_waypoints)
            collision_timing = collision_data.get('collision_timing', 5.0)
            
            logger.info(f"🚗 {vehicle_type} vehicle: {len(animation_waypoints)} points, {collision_timing}s duration")
            
            animate_motion = dwg.animateMotion(
                path=animation_path,
                dur=f"{collision_timing}s",
                repeatCount="1",
                fill="freeze"  # Stop at collision point
            )
            
            vehicle.add(animate_motion)
        
        dwg.add(vehicle)



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
        """Add collision effect animation - FAST VERSION"""
        collision_point = collision_data.get('collision_point', [self.svg_width//2, self.svg_height//2])
        collision_timing = collision_data.get('collision_timing', 4.0)
        
        # Explosion effect
        explosion = dwg.circle(
            center=(collision_point[0], collision_point[1]),
            r=5,
            fill='orange',
            opacity=0
        )
        
        # 🚀 FAST: Explosion starts at 3s, lasts 1.5s
        explosion.add(dwg.animate(
            attributeName='r',
            values='5;50;30',
            dur='1.5s',
            begin='3s',  # Start at 3 seconds
            repeatCount='1'
        ))
        
        explosion.add(dwg.animate(
            attributeName='opacity',
            values='0;1;0.5;0',
            dur='1.5s',
            begin='3s',  # Start at 3 seconds
            repeatCount='1'
        ))
        
        dwg.add(explosion)

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

    def calculate_collision_timing(self, user_path: dict, other_path: dict) -> dict:
        """Calculate when and where vehicles collide - FAST VERSION"""
        logger.info("💥 Calculating collision timing and point - FAST VERSION")
        
        user_waypoints = user_path.get('waypoints', [])
        other_waypoints = other_path.get('waypoints', [])
        
        if not user_waypoints or not other_waypoints:
            return self.create_default_collision()
        
        # Find intersection point of paths
        collision_point = self.find_path_intersection(user_waypoints, other_waypoints)
        
        # 🚀 FAST FIX: Force fast timing instead of calculating from waypoints
        collision_timing = 4.0  # Fixed 4 seconds - fast!
        
        return {
            'collision_point': collision_point,
            'collision_timing': collision_timing,  # Fixed fast timing
            'user_collision_progress': 0.6,
            'other_collision_progress': 0.6,
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
        """Create default collision data when calculation fails - FAST VERSION"""
        return {
            'collision_point': [self.svg_width // 2, self.svg_height // 2],
            'collision_timing': 4.0,  # Fast 4 seconds
            'user_collision_progress': 0.6,
            'other_collision_progress': 0.6,
            'collision_type': 'intersection_collision'
        }


# 🚀 MAIN PROCESS
if __name__ == "__main__":
    logger.info("🧾 Runtime marker | td_rev+image should match deploy | MODE=%s", os.getenv("MODE", ""))
    # 🚨 FORCE LOG TO PROVE NEW CODE IS RUNNING
    logger.info("🔥 NEW CODE VERSION 2.0 - CLEAN ANALYTICS STARTING!")
    logger.info("🔥 This message proves the updated code is deployed!")
    try:
        bucket_name = _get_env("BUCKET_NAME")
        connection_id = _get_env("CONNECTION_ID")
        mode = os.environ.get("MODE", "CLEAN_ANALYTICS")
        
        logger.info("🎉 Starting CLEAN SVG Animation Generation")
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