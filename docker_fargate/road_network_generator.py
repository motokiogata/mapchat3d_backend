# road_network_generator.py

import cv2
import numpy as np
import json
import boto3
import base64
import os
import sys
from io import BytesIO
from PIL import Image
from skimage.morphology import medial_axis, closing, opening, disk, remove_small_objects
from collections import defaultdict
import math
import re
from math import atan2, degrees, sqrt, cos, sin, radians
from shapely.geometry import LineString, Point
from scipy import ndimage
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point
from s3_handler import S3Handler

# AWS Bedrock setup
BEDROCK_MODEL_ID = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_REGION = "ap-northeast-1"

# Input paths (downloaded from S3)
MASK_PATH = "final_road_mask_cleaned.png"
ROADMAP_PATH = "roadmap.png"
SATELLITE_PATH = "satellite.png"

# Enhanced Parameters 
MIN_LINE_LENGTH = 20
CANVAS_SIZE = (1280, 1280)
EDGE_TOLERANCE = 10
LANE_OFFSET_PX = 20
INTERSECTION_RADIUS = 100

# Obstacle analysis parameters
OBSTACLE_MIN_SIZE = 10      
OBSTACLE_MAX_SIZE = 50000   

# Fragment analysis parameters
FRAGMENT_MIN_LENGTH = 30    
FRAGMENT_MAX_LENGTH = 150   

class IntegratedRoadNetworkGenerator:
    def __init__(self, connection_id, bucket_name):
        # AWS S3 setup
        self.connection_id = connection_id
        self.bucket_name = bucket_name
        self.s3_handler = S3Handler(connection_id, bucket_name)
        
        # Core data with consistent IDs
        self.roads = []
        self.intersections = []
        self.lane_trees = []
        
        # ID mapping and consistency
        self.road_id_counter = 0
        self.intersection_id_counter = 0
        self.lane_id_counter = 0
        
        # Cross-reference mappings
        self.road_to_intersections = {}
        self.intersection_to_roads = {}
        self.lane_to_road = {}
        
        # Enhanced metadata - BOTH approaches
        self.bedrock_metadata = {}
        self.comprehensive_metadata = {}
        self.edge_analysis_summary = {}
        self.navigation_graph = {}
        
        # Obstacle analysis data
        self.obstacle_analysis = {}
        self.ignorable_obstacles = []
        self.meaningful_obstacles = []
        
        # Fragment analysis data
        self.fragment_analysis = {}
        self.ignorable_fragments = []
        self.meaningful_fragments = []
        
        # Edge analysis
        self.edge_entry_points = {}
        self.geographic_road_map = {'west': [], 'east': [], 'north': [], 'south': []}
        
        # INTEGRATED: Lane generation data
        self.road_connections = {}

    def get_output_filename(self, base_filename):
        """Generate output filename with connection_id prefix"""
        return f"{self.connection_id}_{base_filename}"

    def upload_output_to_s3(self, local_filename, s3_filename=None):
        """Upload output file to S3"""
        if s3_filename is None:
            s3_filename = local_filename
        
        s3_key = f"outputs/{self.connection_id}/{s3_filename}"
        return self.s3_handler.upload_to_s3(local_filename, s3_key)

    def upload_json_output_to_s3(self, data, s3_filename):
        """Upload JSON data directly to S3"""
        s3_key = f"outputs/{self.connection_id}/{s3_filename}"
        return self.s3_handler.upload_json_to_s3(data, s3_key)

    # ==================== CREATE METADATA ONLY JSONS METHODS ====================

    def extract_metadata_only_from_road(self, road):
        """Extract only metadata and identifiers from road, no coordinate points"""
        metadata = road.get('metadata', {})
        
        return {
            'road_id': road['id'],
            'name': metadata.get('name', f'Road_{road["id"]}'),
            'display_name': metadata.get('display_name', f'Road_{road["id"]}'),
            'road_class': metadata.get('road_class', 'local_street'),
            'road_type': metadata.get('road_type', 'street'),
            'width_category': metadata.get('width_category', 'medium'),
            'estimated_lanes': metadata.get('estimated_lanes', 2),
            'surface_type': metadata.get('surface_type', 'paved'),
            'estimated_speed_limit': metadata.get('estimated_speed_limit', 30),
            'priority': metadata.get('priority', 3),
            'curvature': metadata.get('curvature', 'straight'),
            
            # Navigation-relevant metadata
            'position_flow': metadata.get('position_flow', 'unknown'),
            'simple_direction': metadata.get('simple_direction', 'unknown'),
            'start_area': metadata.get('start_area', 'unknown'),
            'end_area': metadata.get('end_area', 'unknown'),
            
            # Destination and context
            'leads_to': metadata.get('leads_to', {}),
            'can_turn_left': metadata.get('can_turn_left', True),
            'can_turn_right': metadata.get('can_turn_right', True),
            'can_go_straight': metadata.get('can_go_straight', True),
            'has_median': metadata.get('has_median', False),
            'parking_available': metadata.get('parking_available', True),
            
            # Narrative metadata for LLM
            'conversational_identifiers': metadata.get('conversational_identifiers', []),
            'user_likely_descriptions': metadata.get('user_likely_descriptions', []),
            'narrative_visual_characteristics': metadata.get('narrative_visual_characteristics', {}),
            'narrative_directional': metadata.get('narrative_directional', {}),
            'conversation_landmarks': metadata.get('conversation_landmarks', []),
            'route_context': metadata.get('route_context', {}),
            'nearby_landmarks': metadata.get('nearby_landmarks', []),
            'visible_labels': metadata.get('visible_labels', []),
            
            # Edge analysis (position-based, not coordinates)
            'edge_analysis': {
                'has_edge_connection': metadata.get('edge_analysis', {}).get('has_edge_connection', False),
                'edge_sides': metadata.get('edge_analysis', {}).get('edge_sides', []),
                'start_position': metadata.get('edge_analysis', {}).get('start_position', 'unknown'),
                'end_position': metadata.get('edge_analysis', {}).get('end_position', 'unknown')
            }
        }

    def extract_metadata_only_from_intersection(self, intersection):
        """Extract only metadata and identifiers from intersection, no coordinate points"""
        metadata = intersection.get('metadata', {})
        zone = intersection.get('zone', {})
        
        return {
            'intersection_id': intersection['id'],
            'intersection_name': metadata.get('intersection_name', f'Intersection_{intersection["id"]}'),
            'intersection_type': metadata.get('intersection_type', 'cross_intersection'),
            'size_category': metadata.get('size_category', 'medium'),
            'traffic_volume': metadata.get('traffic_volume', 'medium'),
            'complexity': metadata.get('complexity', 'medium'),
            'roads_count': intersection.get('roads_count', 4),
            
            # Infrastructure
            'has_traffic_signals': metadata.get('has_traffic_signals', False),
            'has_crosswalks': metadata.get('has_crosswalks', True),
            'has_turning_lanes': metadata.get('has_turning_lanes', False),
            'estimated_wait_time': metadata.get('estimated_wait_time', 30),
            'navigation_complexity': metadata.get('navigation_complexity', 'medium'),
            
            # Zone information (descriptive, not coordinate-based)
            'zone_type': zone.get('zone_type', 'circular'),
            'zone_radius': zone.get('radius', 80),
            
            # Narrative metadata for LLM
            'conversational_identifiers': metadata.get('conversational_identifiers', []),
            'user_likely_descriptions': metadata.get('user_likely_descriptions', []),
            'navigation_conversations': metadata.get('navigation_conversations', {}),
            'conversation_landmarks': metadata.get('conversation_landmarks', []),
            'route_decision_context': metadata.get('route_decision_context', {}),
            'nearby_landmarks': metadata.get('nearby_landmarks', []),
            'nearby_businesses': metadata.get('nearby_businesses', []),
            
            # Connected roads summary (names and types, not coordinates)
            'connected_roads_summary': metadata.get('connected_roads_summary', {
                'major_roads': [],
                'secondary_roads': [],
                'local_streets': []
            })
        }

    def extract_metadata_only_from_lane(self, lane_tree):
        """Extract only essential lane metadata, no coordinate points"""
        metadata = lane_tree.get('metadata', {})
        branches = lane_tree.get('branches', [])
        
        # Extract branch metadata without coordinates
        branch_metadata = []
        for branch in branches:
            branch_meta = {
                'branch_id': branch.get('branch_id', ''),
                'target_road_id': branch.get('target_road_id', ''),
                'target_road_name': branch.get('target_road_name', ''),
                'target_road_direction': branch.get('target_road_direction', ''),
                'turn_type': branch.get('turn_type', 'unknown'),
                'navigation_instruction': branch.get('navigation_instruction', ''),
                'lht_turn_guidance': branch.get('lht_turn_guidance', ''),
                'is_clean_connection': branch.get('is_clean_connection', False)
            }
            branch_metadata.append(branch_meta)
        
        return {
            'lane_id': lane_tree.get('lane_id', ''),
            'road_id': lane_tree.get('road_id', ''),
            'direction': lane_tree.get('direction', 'unknown'),
            'lane_type': lane_tree.get('lane_type', 'traffic_flow'),
            
            # Essential lane info
            'display_name': metadata.get('display_name', ''),
            'parent_road_name': metadata.get('parent_road_name', ''),
            'traffic_direction': metadata.get('traffic_direction', 'unknown'),
            
            # Road classification inherited from parent
            'road_class': metadata.get('road_class', 'local_street'),
            'estimated_speed_limit': metadata.get('estimated_speed_limit', 30),
            'priority': metadata.get('priority', 3),
            
            # LHT-specific usage guidance
            'lht_lane_usage': metadata.get('lht_lane_usage', {}),
            
            # Branches (without coordinates)
            'branches_metadata': branch_metadata,
            'branch_count': len(branches),
            
            # Connection information
            'intersection_connection': lane_tree.get('intersection_connection', {})
        }

    def save_metadata_only_outputs(self):
        """Save metadata-only JSON files for LLM consumption"""
        print("💾 SAVING METADATA-ONLY JSON FILES FOR LLM...")
        
        # 1. Extract roads metadata only
        roads_metadata = []
        for road in self.roads:
            road_meta = self.extract_metadata_only_from_road(road)
            roads_metadata.append(road_meta)
        
        # 2. Extract intersections metadata only
        intersections_metadata = []
        for intersection in self.intersections:
            intersection_meta = self.extract_metadata_only_from_intersection(intersection)
            intersections_metadata.append(intersection_meta)
        
        # 3. Extract lanes metadata only
        lanes_metadata = []
        for lane_tree in self.lane_trees:
            lane_meta = self.extract_metadata_only_from_lane(lane_tree)
            lanes_metadata.append(lane_meta)
        
        # 4. Create comprehensive metadata summary
        comprehensive_metadata = getattr(self, 'comprehensive_metadata', {})
        narrative_metadata = getattr(self, 'narrative_metadata', {})
        obstacle_analysis = getattr(self, 'obstacle_analysis', {})
        fragment_analysis = getattr(self, 'fragment_analysis', {})
        
        # Count narrative elements
        total_conv_ids = sum(len(road.get('conversational_identifiers', [])) for road in roads_metadata)
        total_user_descriptions = sum(len(road.get('user_likely_descriptions', [])) for road in roads_metadata)
        total_landmarks = set()
        
        for road in roads_metadata:
            for landmark in road.get('conversation_landmarks', []):
                if isinstance(landmark, dict):
                    total_landmarks.add(landmark.get('name', str(landmark)))
                else:
                    total_landmarks.add(str(landmark))
        
        # 5. Save individual metadata files
        
        # Roads metadata only
        roads_only_data = {
            'roads_metadata': roads_metadata,
            'metadata_summary': {
                'total_roads': len(roads_metadata),
                'road_classes': list(set(road.get('road_class', 'unknown') for road in roads_metadata)),
                'conversational_identifiers_count': sum(len(road.get('conversational_identifiers', [])) for road in roads_metadata),
                'user_descriptions_count': sum(len(road.get('user_likely_descriptions', [])) for road in roads_metadata),
                'has_narrative_metadata': total_conv_ids > 0
            },
            'coordinate_system': 'metadata_only',
            'for_llm_consumption': True
        }
        
        roads_only_data = self.make_json_serializable(roads_only_data)
        roads_filename = self.get_output_filename("roads_metadata_only.json")
        with open(roads_filename, 'w') as f:
            json.dump(roads_only_data, f, indent=2)
        self.upload_output_to_s3(roads_filename)
        
        # Intersections metadata only
        intersections_only_data = {
            'intersections_metadata': intersections_metadata,
            'metadata_summary': {
                'total_intersections': len(intersections_metadata),
                'intersection_types': list(set(i.get('intersection_type', 'unknown') for i in intersections_metadata)),
                'traffic_volumes': list(set(i.get('traffic_volume', 'unknown') for i in intersections_metadata)),
                'has_narrative_metadata': sum(len(i.get('conversational_identifiers', [])) for i in intersections_metadata) > 0
            },
            'coordinate_system': 'metadata_only',
            'for_llm_consumption': True
        }
        
        intersections_only_data = self.make_json_serializable(intersections_only_data)
        intersections_filename = self.get_output_filename("intersections_metadata_only.json")
        with open(intersections_filename, 'w') as f:
            json.dump(intersections_only_data, f, indent=2)
        self.upload_output_to_s3(intersections_filename)
        
        # Lanes metadata only
        lanes_only_data = {
            'lanes_metadata': lanes_metadata,
            'metadata_summary': {
                'total_lanes': len(lanes_metadata),
                'traffic_directions': list(set(lane.get('direction', 'unknown') for lane in lanes_metadata)),
                'lane_types': list(set(lane.get('lane_type', 'unknown') for lane in lanes_metadata)),
                'total_branches': sum(lane.get('branch_count', 0) for lane in lanes_metadata),
                'lanes_with_branches': sum(1 for lane in lanes_metadata if lane.get('branch_count', 0) > 0),
                'traffic_system': lanes_metadata[0].get('traffic_system', 'unknown') if lanes_metadata else 'unknown',
                'has_narrative_metadata': sum(len(lane.get('conversational_identifiers', [])) for lane in lanes_metadata) > 0,
                'has_traffic_geometry': sum(1 for lane in lanes_metadata if lane.get('traffic_system', 'unknown') != 'unknown') > 0
            },
            'coordinate_system': 'metadata_only',
            'for_llm_consumption': True
        }
        
        lanes_only_data = self.make_json_serializable(lanes_only_data)
        lanes_filename = self.get_output_filename("lanes_metadata_only.json")
        with open(lanes_filename, 'w') as f:
            json.dump(lanes_only_data, f, indent=2)
        self.upload_output_to_s3(lanes_filename)
        
        # 6. Save comprehensive metadata-only file
        comprehensive_metadata_only = {
            'network_summary': {
                'total_roads': len(roads_metadata),
                'total_intersections': len(intersections_metadata),
                'total_lanes': len(lanes_metadata),
                'total_branches': sum(lane.get('branch_count', 0) for lane in lanes_metadata)
            },
            
            'roads_metadata': roads_metadata,
            'intersections_metadata': intersections_metadata,
            'lanes_metadata': lanes_metadata,
            
            # LLM-friendly narrative summary
            'narrative_summary': {
                'total_conversational_identifiers': total_conv_ids,
                'total_user_descriptions': total_user_descriptions,
                'unique_landmarks': list(total_landmarks),
                'area_context': narrative_metadata.get('area_context', {}),
                'common_route_patterns': narrative_metadata.get('common_route_patterns', []),
                'road_narratives_count': len(narrative_metadata.get('road_narratives', [])),
                'intersection_narratives_count': len(narrative_metadata.get('intersection_narratives', []))
            },
            
            # Visual analysis summary (no coordinates)
            'visual_analysis_summary': {
                'road_labels_found': len(comprehensive_metadata.get('road_labels', [])),
                'landmarks_identified': len(comprehensive_metadata.get('landmarks', [])),
                'road_classifications_made': len(comprehensive_metadata.get('road_classifications', [])),
                'directional_context_available': bool(comprehensive_metadata.get('directional_context', {}))
            },
            
            # Processing summary
            'processing_summary': {
                'obstacles_removed': len(getattr(self, 'ignorable_obstacles', [])),
                'fragments_cleaned': len(getattr(self, 'ignorable_fragments', [])),
                'roads_trimmed_at_intersections': sum(1 for road in self.roads if road.get('trimmed_from_intersection')),
                'has_comprehensive_metadata': True,
                'has_narrative_metadata': total_conv_ids > 0,
                'metadata_version': '2.0_metadata_only'
            },
            
            'metadata': {
                'coordinate_system': 'metadata_only',
                'contains_coordinates': False,
                'for_llm_consumption': True,
                'supports_narrative_parsing': True,
                'supports_landmark_navigation': True,
                'supports_turn_by_turn': True,
                'file_type': 'metadata_only_network_description'
            }
        }
        
        comprehensive_metadata_only = self.make_json_serializable(comprehensive_metadata_only)
        comprehensive_filename = self.get_output_filename("metadata_only_network.json")
        with open(comprehensive_filename, 'w') as f:
            json.dump(comprehensive_metadata_only, f, indent=2)
        self.upload_output_to_s3(comprehensive_filename)
        
        print(f"  💾 Saved comprehensive metadata-only to {comprehensive_filename}")
        print(f"  💾 Saved roads metadata-only to {roads_filename}")
        print(f"  💾 Saved intersections metadata-only to {intersections_filename}")
        print(f"  💾 Saved lanes metadata-only to {lanes_filename}")
        
        print(f"\n📊 METADATA-ONLY FILES SUMMARY:")
        print(f"  🛣️  Roads: {len(roads_metadata)} with {total_conv_ids} conversation IDs")
        print(f"  🚦 Intersections: {len(intersections_metadata)} with full context")
        print(f"  🛤️  Lanes: {len(lanes_metadata)} with traffic geometry")
        print(f"  💬 Narrative: {total_user_descriptions} user descriptions, {len(total_landmarks)} landmarks")
        print(f"  📏 File sizes: Compressed for LLM consumption (no coordinate arrays)")

    # ==================== CLEANING & PREPROCESSING METHODS ====================

    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for Bedrock API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def find_obstacles_in_roads(self, mask_path):
        """Find black obstacles for LLM analysis"""
        print("🔍 FINDING ALL BLACK OBJECTS FOR LLM ANALYSIS...")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        
        debug_mask_before = self.get_output_filename("debug_mask_before_obstacle_removal.png")
        cv2.imwrite(debug_mask_before, mask)
        self.upload_output_to_s3(debug_mask_before)
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        inverted_mask = cv2.bitwise_not(binary_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
        
        obstacles = []
        print(f"  🔍 Found {num_labels-1} black regions total")
        
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            
            if area < OBSTACLE_MAX_SIZE:
                obstacle_mask = (labels == i).astype(np.uint8)
                center_x = int(centroids[i][0])
                center_y = int(centroids[i][1])
                aspect_ratio = float(max(w, h) / min(w, h)) if min(w, h) > 0 else 1.0
                is_elongated = bool(aspect_ratio > 3)
                
                obstacles.append({
                    'id': len(obstacles),
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'is_elongated': is_elongated,
                    'component_label': i,
                    'obstacle_mask': obstacle_mask,
                    'original_labels': labels,
                    'binary_mask': binary_mask
                })
        
        return obstacles

    def analyze_obstacles_with_bedrock(self, obstacles, mask_path, roadmap_path, satellite_path):
        """Use Bedrock to classify obstacles"""
        if not obstacles:
            return {"ignorable_obstacles": [], "meaningful_obstacles": []}
        
        print("🧠 ANALYZING OBSTACLES WITH CLAUDE...")
        
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        
        mask_b64 = self.encode_image_to_base64(mask_path)
        roadmap_b64 = self.encode_image_to_base64(roadmap_path)
        satellite_b64 = self.encode_image_to_base64(satellite_path)
        
        obstacle_summary = []
        for obs in obstacles:
            obstacle_summary.append({
                "id": int(obs['id']),
                "center": [int(obs['center'][0]), int(obs['center'][1])],
                "area": int(obs['area']),
                "bbox": [int(obs['bbox'][0]), int(obs['bbox'][1]), int(obs['bbox'][2]), int(obs['bbox'][3])],
                "aspect_ratio": float(obs['aspect_ratio']),
                "is_elongated": bool(obs['is_elongated'])
            })
        
        prompt = f"""
        TASK: Analyze black objects within white road areas to determine which should be removed for clean centerline generation.

        DETECTED BLACK OBJECTS: {json.dumps(obstacle_summary, indent=2)}

        CLASSIFICATION: For each object ID, classify as IGNORABLE (lane separators) or MEANINGFUL (real objects).

        Return ONLY this JSON format:
        {{
          "analysis_summary": "Brief description",
          "ignorable_obstacles": [
            {{"id": 0, "reason": "Lane separator marking"}}
          ],
          "meaningful_obstacles": [
            {{"id": 1, "reason": "Building structure"}}
          ]
        }}
        """
        
        try:
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 3000,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": mask_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": roadmap_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": satellite_b64}}
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                return json.loads(json_text)
            
        except Exception as e:
            print(f"  💥 Bedrock obstacle analysis failed: {e}")
        
        return {"error": "Analysis failed"}

    def remove_ignorable_obstacles(self, mask_path, obstacles, ignorable_obstacle_ids):
        """Remove LLM-classified ignorable obstacles from mask"""
        print("🧹 REMOVING LLM-CLASSIFIED IGNORABLE OBSTACLES...")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cleaned_mask = mask.copy()
        
        for obs_id in ignorable_obstacle_ids:
            obstacle = next((obs for obs in obstacles if obs['id'] == obs_id), None)
            if obstacle:
                obstacle_mask = obstacle['obstacle_mask']
                cleaned_mask[obstacle_mask == 1] = 255
                print(f"  🗑️  Removed obstacle {obs_id}")
        
        debug_mask_after = self.get_output_filename("debug_mask_after_obstacle_removal.png")
        cv2.imwrite(debug_mask_after, cleaned_mask)
        self.upload_output_to_s3(debug_mask_after)
        
        return cleaned_mask


    ############# INTEGRATED SMART OBSTACLE REMOVAL PIPELINE #############
    def smart_obstacle_removal_pipeline(self, mask_path):
        print("\n🎯 LLM-POWERED SMART OBSTACLE REMOVAL PIPELINE")
        print("=" * 60)
        
        obstacles = self.find_obstacles_in_roads(mask_path)
        
        if not obstacles:
            # ... existing code
            cleaned_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return cleaned_mask
    
        print(f"🧠 Found {len(obstacles)} obstacles, sending to LLM...")
        
        classification = self.analyze_obstacles_with_bedrock(obstacles, mask_path, ROADMAP_PATH, SATELLITE_PATH)
        
        # 🔥 DEBUG THE CLASSIFICATION RESULT
        print(f"🔍 LLM CLASSIFICATION RESULT:")
        print(f"   Raw classification type: {type(classification)}")
        print(f"   Raw classification: {classification}")
        
        self.obstacle_analysis = classification
        
        if "error" in classification:
            print("⚠️ LLM FAILED - Using fallback logic")
            ignorable_ids = [obs['id'] for obs in obstacles if obs['area'] < 5000 and obs['is_elongated']]
            print(f"🔧 Fallback found {len(ignorable_ids)} ignorable obstacles")
            self.ignorable_obstacles = [{"id": obs_id, "reason": "Fallback: small/elongated"} for obs_id in ignorable_ids]
            self.meaningful_obstacles = [{"id": obs['id'], "reason": "Fallback: preserved"} for obs in obstacles if obs['id'] not in ignorable_ids]
        else:
            print("✅ LLM SUCCESS - Using LLM classification")
            ignorable_ids = [obs['id'] for obs in classification.get('ignorable_obstacles', [])]
            print(f"🧠 LLM found {len(ignorable_ids)} ignorable obstacles")
            print(f"🧠 Ignorable obstacle details: {classification.get('ignorable_obstacles', [])}")
            self.ignorable_obstacles = classification.get('ignorable_obstacles', [])
            self.meaningful_obstacles = classification.get('meaningful_obstacles', [])
        
        print(f"🗑️ FINAL DECISION: Will remove {len(ignorable_ids)} obstacles")
        print(f"🗑️ Obstacle IDs to remove: {ignorable_ids}")
        
        if ignorable_ids:
            print("🧹 Calling remove_ignorable_obstacles...")
            cleaned_mask = self.remove_ignorable_obstacles(mask_path, obstacles, ignorable_ids)
            print("✅ Obstacle removal completed")
        else:
            print("⚠️ NO OBSTACLES TO REMOVE - returning original mask")
            cleaned_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            debug_mask_after = self.get_output_filename("debug_mask_after_obstacle_removal.png")
            cv2.imwrite(debug_mask_after, cleaned_mask)
            self.upload_output_to_s3(debug_mask_after)
        
        return cleaned_mask



    def analyze_expected_road_and_intersection_count_with_bedrock(self, cleaned_mask_path, roadmap_path, satellite_path):
        """Enhanced network structure analysis"""
        print("🧠 ANALYZING EXPECTED ROAD AND INTERSECTION COUNT WITH CLAUDE...")
        
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        mask_b64 = self.encode_image_to_base64(cleaned_mask_path)
        roadmap_b64 = self.encode_image_to_base64(roadmap_path)
        satellite_b64 = self.encode_image_to_base64(satellite_path)
        
        prompt = """
        TASK: Analyze these images to determine how many MAIN ROADS and MAJOR INTERSECTIONS there should be.

        You have 3 images:
        1. ROAD MASK: Black/white where WHITE = roads, BLACK = non-roads
        2. ROADMAP: Normal road map with labels
        3. SATELLITE: Aerial view

        Please analyze the road network structure and count:

        MAIN ROADS: Distinct road segments that extend across the area
        - Count major routes, not every small segment
        - Look for continuous roads that serve as primary traffic routes

        MAJOR INTERSECTIONS: Significant points where main roads meet
        - Count distinct intersection AREAS, not every junction point
        - A single intersection area where 4 roads meet = 1 intersection (not 4)
        - Focus on major decision points for navigation

        Looking at the mask, I can see this appears to be a cross-pattern intersection.

        Return ONLY this JSON format:
        {
        "analysis": "Brief description of the road network pattern",
        "expected_main_roads": 4,
        "expected_major_intersections": 1,
        "intersection_pattern": "4-way_cross",
        "road_pattern": "cross_intersection",
        "confidence": "high"
        }
        """
        
        try:
            print("  🔍 Asking Claude to analyze road network structure...")
            
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1500,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": mask_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": roadmap_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": satellite_b64}}
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                try:
                    network_analysis = json.loads(json_text)
                    expected_roads = network_analysis.get('expected_main_roads', 4)
                    expected_intersections = network_analysis.get('expected_major_intersections', 1)
                    confidence = network_analysis.get('confidence', 'medium')
                    
                    print(f"  📊 LLM Network Analysis:")
                    print(f"    🛣️  Expected main roads: {expected_roads}")
                    print(f"    🚦 Expected major intersections: {expected_intersections}")
                    print(f"    📋 Road pattern: {network_analysis.get('road_pattern', 'unknown')}")
                    print(f"    🚦 Intersection pattern: {network_analysis.get('intersection_pattern', 'unknown')}")
                    print(f"    ✅ Confidence: {confidence}")
                    
                    return network_analysis
                    
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parsing failed: {e}")
                    return {"expected_main_roads": 4, "expected_major_intersections": 1, "confidence": "fallback"}
            
        except Exception as e:
            print(f"  💥 Network analysis failed: {e}")
        
        return {"expected_main_roads": 4, "expected_major_intersections": 1, "confidence": "fallback"}

    # ==================== SKELETON PROCESSING METHODS ====================
    
    def extract_medial_axis(self, cleaned_mask_image):
        """Extract skeleton with better preprocessing"""
        print("📍 EXTRACTING MEDIAL AXIS WITH BETTER PREPROCESSING...")
        
        # 🔥 FIX: Handle both numpy array and file path inputs
        if isinstance(cleaned_mask_image, str):
            img = cv2.imread(cleaned_mask_image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image loading failed from {cleaned_mask_image}")
        else:
            # It's already a numpy array
            img = cleaned_mask_image
            
        if img is None:
            raise FileNotFoundError(f"Image loading failed")
        
        # IMPROVED: Better preprocessing to reduce fragmentation
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Fill small holes to reduce fragmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        binary = binary // 255  # Convert to 0,1
        skel = medial_axis(binary).astype(np.uint8)
        
        debug_skeleton = self.get_output_filename("debug_skeleton.png")
        cv2.imwrite(debug_skeleton, skel * 255)
        self.upload_output_to_s3(debug_skeleton)
        print(f"✅ Initial skeleton extracted - points: {np.sum(skel)}")
        
        return skel

    def find_skeleton_fragments(self, skeleton):
        """Find all skeleton fragments for LLM analysis"""
        print("🔍 FINDING SKELETON FRAGMENTS FOR LLM ANALYSIS...")
        
        visited = np.zeros_like(skeleton, dtype=bool)
        fragments = []
        
        for y in range(skeleton.shape[0]):
            for x in range(skeleton.shape[1]):
                if skeleton[y, x] == 1 and not visited[y, x]:
                    # Trace this fragment
                    fragment_points = self.trace_fragment(skeleton, y, x, visited)
                    
                    if len(fragment_points) > 5:  # Minimum fragment size
                        fragments.append({
                            'id': len(fragments),
                            'points': fragment_points,
                            'length': len(fragment_points),
                            'start': fragment_points[0],
                            'end': fragment_points[-1],
                            'is_short': len(fragment_points) < FRAGMENT_MIN_LENGTH,
                            'bbox': self.calculate_fragment_bbox(fragment_points)
                        })
        
        print(f"  🔍 Found {len(fragments)} skeleton fragments")
        
        # Create debug visualization
        debug_img = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, fragment in enumerate(fragments):
            color = colors[i % len(colors)]
            points = fragment['points']
            
            for j in range(len(points) - 1):
                pt1 = (int(points[j][0]), int(points[j][1]))
                pt2 = (int(points[j+1][0]), int(points[j+1][1]))
                cv2.line(debug_img, pt1, pt2, color, 2)
            
            # Mark fragment number
            if points:
                mid_pt = points[len(points)//2]
                cv2.putText(debug_img, f"F{fragment['id']}({fragment['length']})", 
                           (int(mid_pt[0]), int(mid_pt[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        debug_fragments = self.get_output_filename("debug_fragments_analysis.png")
        cv2.imwrite(debug_fragments, debug_img)
        self.upload_output_to_s3(debug_fragments)
        print(f"  💾 Saved fragment analysis to {debug_fragments}")
        
        return fragments

    def trace_fragment(self, skeleton, start_y, start_x, visited):
        """Trace a single fragment"""
        points = []
        queue = [(start_y, start_x)]
        
        while queue:
            y, x = queue.pop(0)
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            points.append((x, y))  # Note: x, y order for consistency
            
            # Find unvisited neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] == 1 and not visited[ny, nx]):
                        queue.append((ny, nx))
        
        return points

    def calculate_fragment_bbox(self, points):
        """Calculate bounding box for fragment"""
        if not points:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def analyze_fragments_with_bedrock(self, fragments, expected_road_count=None):
        """Enhanced fragment analysis with expected road count"""
        if not fragments:
            return {"ignorable_fragments": [], "meaningful_fragments": []}
        
        print("🧠 ANALYZING SKELETON FRAGMENTS WITH CLAUDE...")
        
        # Filter fragments that are candidates for removal
        candidate_fragments = [f for f in fragments if f['length'] < FRAGMENT_MAX_LENGTH]
        
        if not candidate_fragments:
            print("  ℹ️  No candidate fragments found")
            return {"ignorable_fragments": [], "meaningful_fragments": fragments}
        
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        
        debug_fragments = self.get_output_filename("debug_fragments_analysis.png")
        mask_b64 = self.encode_image_to_base64(debug_fragments)
        
        fragment_summary = []
        for frag in candidate_fragments:
            fragment_summary.append({
                "id": int(frag['id']),
                "length": int(frag['length']),
                "start": [int(frag['start'][0]), int(frag['start'][1])],
                "end": [int(frag['end'][0]), int(frag['end'][1])],
                "bbox": [int(x) for x in frag['bbox']],
                "is_short": bool(frag['is_short'])
            })
        
        # Create expected road count context
        expected_context = ""
        if expected_road_count:
            total_fragments = len(fragments)
            target_removal = max(0, total_fragments - expected_road_count)
            expected_context = f"""
            
    IMPORTANT GUIDANCE:
    - Based on road mask analysis, there should be approximately {expected_road_count} main roads
    - Current total fragments: {total_fragments}
    - To reach target, consider removing approximately {target_removal} fragments
    - Prioritize keeping the {expected_road_count} longest and most connected fragments
    - Focus on removing short, isolated, disconnected fragments first
            """
        else:
            expected_context = ""
        
        prompt = f"""
        TASK: Analyze skeleton fragments to identify which are NOISE vs REAL ROADS.

        You're looking at a skeleton image where different colored lines represent different fragments.
        The goal is to keep only MAIN ROAD CENTERLINES and remove fragmentary noise.

        DETECTED FRAGMENTS: {json.dumps(fragment_summary, indent=2)}
        {expected_context}

        CLASSIFICATION RULES:
        **IGNORABLE** (should be REMOVED):
        - Very short fragments (< 30 pixels) that don't connect main roads
        - Isolated small fragments that are clearly noise
        - Fragments that appear to be artifacts from skeleton extraction
        - Short dead-end spurs that don't represent real roads
        - Disconnected tiny segments that break the main road flow

        **MEANINGFUL** (should be PRESERVED):
        - Long fragments that clearly represent main roads
        - Fragments that connect to other fragments (part of main network)
        - Fragments that align with the main road structure
        - Even shorter fragments if they're clearly part of the main road network
        - Fragments that form the core road intersection structure

        STRATEGY:
        1. Identify the longest fragments - these are likely main roads
        2. Remove very short isolated fragments first
        3. Keep fragments that maintain network connectivity
        4. Aim for clean, continuous road centerlines

        Return ONLY this JSON format:
        {{
        "analysis_summary": "Found X main roads and Y noise fragments, targeting Z final roads",
        "ignorable_fragments": [
            {{"id": 0, "reason": "Short isolated fragment - likely noise"}},
            {{"id": 1, "reason": "Disconnected spur - artifact"}}
        ],
        "meaningful_fragments": [
            {{"id": 2, "reason": "Main road centerline - long and connected"}},
            {{"id": 3, "reason": "Core intersection connector"}}
        ]
        }}
        """
        
        try:
            print("  🔍 Sending enhanced fragment analysis to Claude...")
            if expected_road_count:
                print(f"      🎯 Target: {expected_road_count} main roads")
            
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": mask_b64}}
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                try:
                    fragment_classification = json.loads(json_text)
                    
                    ignorable_count = len(fragment_classification.get('ignorable_fragments', []))
                    meaningful_count = len(fragment_classification.get('meaningful_fragments', []))
                    
                    print(f"  📊 LLM Fragment Classification:")
                    print(f"    🗑️  Ignorable fragments: {ignorable_count}")
                    print(f"    🛣️  Meaningful fragments: {meaningful_count}")
                    
                    return fragment_classification
                    
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parsing failed: {e}")
                    return {"error": "JSON parse failed", "raw_response": analysis_text}
            
        except Exception as e:
            print(f"  💥 Bedrock fragment analysis failed: {e}")
            return {"error": str(e)}

    def remove_ignorable_fragments(self, skeleton, fragments, ignorable_fragment_ids):
        """Remove LLM-classified ignorable fragments from skeleton"""
        print("🧹 REMOVING LLM-CLASSIFIED IGNORABLE FRAGMENTS...")
        
        cleaned_skeleton = skeleton.copy()
        removed_count = 0
        
        for frag_id in ignorable_fragment_ids:
            fragment = next((f for f in fragments if f['id'] == frag_id), None)
            if fragment is None:
                continue
            
            # Remove all points of this fragment
            for point in fragment['points']:
                x, y = int(point[0]), int(point[1])
                if 0 <= y < cleaned_skeleton.shape[0] and 0 <= x < cleaned_skeleton.shape[1]:
                    cleaned_skeleton[y, x] = 0
            
            removed_count += 1
            print(f"  🗑️  Removed fragment {frag_id} (length: {fragment['length']})")
        
        print(f"  ✅ Removed {removed_count} ignorable fragments")
        
        debug_skeleton_cleaned = self.get_output_filename("debug_skeleton_after_fragment_removal.png")
        cv2.imwrite(debug_skeleton_cleaned, cleaned_skeleton * 255)
        self.upload_output_to_s3(debug_skeleton_cleaned)
        print(f"  💾 Saved cleaned skeleton to {debug_skeleton_cleaned}")
        
        return cleaned_skeleton

    def smart_fragment_cleanup_pipeline(self, skeleton, expected_road_count=None):
        """LLM-powered fragment cleanup with expected road count guidance"""
        print("\n🧹 LLM-POWERED FRAGMENT CLEANUP PIPELINE")
        print("=" * 50)
        
        # Step 1: Find all skeleton fragments
        fragments = self.find_skeleton_fragments(skeleton)
        
        # Use expected road count if provided
        if expected_road_count:
            target_fragments = expected_road_count
            tolerance = max(2, expected_road_count // 2)  # Allow some flexibility
            reasonable_range = (target_fragments, target_fragments + tolerance)
            
            print(f"  🎯 Expected roads: {expected_road_count}")
            print(f"  📊 Target fragment range: {reasonable_range[0]}-{reasonable_range[1]}")
            
            if reasonable_range[0] <= len(fragments) <= reasonable_range[1]:
                print(f"✅ Fragment count ({len(fragments)}) matches expected range - minimal cleanup")
                # Only remove very tiny fragments
                tiny_threshold = 10
                ignorable_ids = [f['id'] for f in fragments if f['length'] < tiny_threshold]
                if ignorable_ids:
                    print(f"  🔧 Removing {len(ignorable_ids)} tiny fragments (< {tiny_threshold} pixels)")
                    return self.remove_ignorable_fragments(skeleton, fragments, ignorable_ids)
                return skeleton
        else:
            # Fallback to adaptive approach
            reasonable_threshold = max(6, min(15, len(fragments) // 3))
            if len(fragments) <= reasonable_threshold:
                print(f"✅ Fragment count ({len(fragments)}) is reasonable - skipping cleanup")
                return skeleton
        
        # Step 2: Analyze fragments with LLM (with expected count context)
        print(f"\n🧠 STEP 2: Analyzing {len(fragments)} fragments with Claude...")
        classification = self.analyze_fragments_with_bedrock(fragments, expected_road_count)
        
        # Store analysis results
        self.fragment_analysis = classification
        
        if "error" in classification:
            print("⚠️  LLM fragment analysis failed - using guided fallback")
            if expected_road_count:
                # More aggressive removal to reach target
                target_removal = len(fragments) - expected_road_count
                fragments_by_length = sorted(fragments, key=lambda f: f['length'])
                ignorable_ids = [f['id'] for f in fragments_by_length[:target_removal]]
                print(f"  🔧 Guided removal: targeting {expected_road_count} roads")
            else:
                # Conservative fallback
                ignorable_ids = [f['id'] for f in fragments if f['length'] < 20]
            
            self.ignorable_fragments = [{"id": frag_id, "reason": "Guided/fallback removal"} for frag_id in ignorable_ids]
            self.meaningful_fragments = [{"id": f['id'], "reason": "Preserved"} for f in fragments if f['id'] not in ignorable_ids]
        else:
            # Use LLM classification
            ignorable_ids = [frag['id'] for frag in classification.get('ignorable_fragments', [])]
            self.ignorable_fragments = classification.get('ignorable_fragments', [])
            self.meaningful_fragments = classification.get('meaningful_fragments', [])
        
        # Step 3: Remove ignorable fragments
        print("\n🗑️  STEP 3: Removing ignorable fragments...")
        if ignorable_ids:
            cleaned_skeleton = self.remove_ignorable_fragments(skeleton, fragments, ignorable_ids)
            remaining_fragments = len(fragments) - len(ignorable_ids)
            print(f"  ✅ Reduced from {len(fragments)} to {remaining_fragments} fragments")
        else:
            print("ℹ️  No ignorable fragments to remove per LLM analysis")
            cleaned_skeleton = skeleton
        
        return cleaned_skeleton

    # ==================== ROAD TRACING METHODS ====================
    
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
        """Find endpoints and junctions with better detection"""
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
                    elif degree >= 3:
                        junctions.append((y, x))
        
        # Filter out junction points that are too close to each other
        if len(junctions) > 1:
            filtered_junctions = []
            used = set()
            
            for i, (y1, x1) in enumerate(junctions):
                if (y1, x1) in used:
                    continue
                
                # Find nearby junctions
                cluster = [(y1, x1)]
                for j, (y2, x2) in enumerate(junctions):
                    if i != j and (y2, x2) not in used:
                        dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                        if dist <= 20:  # Cluster nearby junction points
                            cluster.append((y2, x2))
                            used.add((y2, x2))
                
                # Take the centroid of the cluster
                center_y = int(np.mean([p[0] for p in cluster]))
                center_x = int(np.mean([p[1] for p in cluster]))
                filtered_junctions.append((center_y, center_x))
                used.add((y1, x1))
            
            junctions = filtered_junctions
        
        print(f"✅ Found {len(endpoints)} endpoints and {len(junctions)} junctions")
        return endpoints, junctions

    def trace_path_to_intersection(self, skeleton, start_y, start_x, visited, junctions):
        """Trace path from endpoint towards intersection"""
        path = [(start_x, start_y)]
        visited[start_y, start_x] = True
        
        current_y, current_x = start_y, start_x
        junction_set = set(junctions)
        
        while True:
            neighbors = self.get_neighbors_8(current_y, current_x, skeleton)
            unvisited_neighbors = [(ny, nx) for ny, nx in neighbors if not visited[ny, nx]]
            
            if len(unvisited_neighbors) == 0:
                break
            elif len(unvisited_neighbors) == 1:
                next_y, next_x = unvisited_neighbors[0]
                
                # Check if we're approaching the intersection area
                if (next_y, next_x) in junction_set:
                    # We've reached the intersection - add this point and stop
                    path.append((next_x, next_y))
                    visited[next_y, next_x] = True
                    break
                else:
                    # Continue normal tracing
                    visited[next_y, next_x] = True
                    path.append((next_x, next_y))
                    current_y, current_x = next_y, next_x
            else:
                # Multiple neighbors - we're at a junction
                if junction_set:
                    best_neighbor = self.choose_best_neighbor_toward_intersection(
                        current_y, current_x, unvisited_neighbors, junction_set)
                    if best_neighbor:
                        next_y, next_x = best_neighbor
                        visited[next_y, next_x] = True
                        path.append((next_x, next_y))
                        current_y, current_x = next_y, next_x
                    else:
                        break
                else:
                    break
        
        return path

    def choose_best_neighbor_toward_intersection(self, current_y, current_x, neighbors, junction_set):
        """Choose neighbor that leads toward intersection center"""
        if not junction_set:
            return neighbors[0] if neighbors else None
        
        # Calculate center of junction area
        junction_center_y = np.mean([j[0] for j in junction_set])
        junction_center_x = np.mean([j[1] for j in junction_set])
        
        best_neighbor = None
        min_distance = float('inf')
        
        for ny, nx in neighbors:
            # Distance from this neighbor to junction center
            dist = np.sqrt((ny - junction_center_y)**2 + (nx - junction_center_x)**2)
            if dist < min_distance:
                min_distance = dist
                best_neighbor = (ny, nx)
        
        return best_neighbor

    def trace_component_simple(self, skeleton, start_y, start_x, visited):
        """Simple component tracing"""
        component = []
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            component.append((x, y))  # Note: x, y order
            
            # Add unvisited neighbors to stack
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] == 1 and not visited[ny, nx]):
                        stack.append((ny, nx))
        
        return component

    def find_remaining_components(self, skeleton, visited):
        """Find any remaining unvisited skeleton components"""
        components = []
        temp_visited = visited.copy()
        
        for y in range(skeleton.shape[0]):
            for x in range(skeleton.shape[1]):
                if skeleton[y, x] == 1 and not temp_visited[y, x]:
                    # Found unvisited skeleton point - trace this component
                    component = self.trace_component_simple(skeleton, y, x, temp_visited)
                    if len(component) > 5:  # Minimum component size
                        components.append(component)
        
        return components

    def trace_skeleton_paths(self, skeleton):
        """Proper road tracing that respects intersection structure"""
        visited = np.zeros_like(skeleton, dtype=bool)
        endpoints, junctions = self.find_endpoints_and_junctions(skeleton)
        
        roads = []
        
        print(f"🛣️  TRACING PATHS: {len(endpoints)} endpoints, {len(junctions)} junctions")
        
        # STEP 1: Find intersection center (if it's a cross pattern)
        intersection_center = None
        if junctions:
            # Find the most central junction point as intersection center
            center_y = int(np.mean([j[0] for j in junctions]))
            center_x = int(np.mean([j[1] for j in junctions]))
            intersection_center = (center_y, center_x)
            print(f"  🎯 Detected intersection center at ({center_x}, {center_y})")
        
        # STEP 2: Trace from each endpoint to intersection center
        for i, (start_y, start_x) in enumerate(endpoints):
            if visited[start_y, start_x]:
                continue
            
            # Trace from endpoint towards intersection
            path = self.trace_path_to_intersection(skeleton, start_y, start_x, visited, junctions)
            
            if len(path) >= MIN_LINE_LENGTH:
                roads.append({
                    "id": self.road_id_counter,
                    "points": path,
                    "start_type": "endpoint_to_intersection",
                    "length": len(path)
                })
                self.road_id_counter += 1
                print(f"  ✅ Road {self.road_id_counter-1}: endpoint ({start_x}, {start_y}) to intersection ({len(path)} points)")
        
        # STEP 3: Handle any remaining unvisited skeleton parts
        remaining_components = self.find_remaining_components(skeleton, visited)
        
        for component in remaining_components:
            if len(component) >= MIN_LINE_LENGTH:
                roads.append({
                    "id": self.road_id_counter,
                    "points": component,
                    "start_type": "remaining_component",
                    "length": len(component)
                })
                self.road_id_counter += 1
                print(f"  ✅ Road {self.road_id_counter-1}: remaining component ({len(component)} points)")
        
        print(f"  📊 Total roads traced: {len(roads)}")
        return roads, endpoints, junctions

    # ==================== INTERSECTION METHODS ====================
    
    def validate_intersection_position(self, center):
        """Validate intersection position is not too close to boundaries"""
        center_x, center_y = center
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        
        margin = 100
        min_x, max_x = margin, w - margin
        min_y, max_y = margin, h - margin
        
        safe_x = max(min_x, min(max_x, center_x))
        safe_y = max(min_y, min(max_y, center_y))
        
        if (safe_x, safe_y) != (center_x, center_y):
            print(f"    ⚠️  Moved from edge: ({center_x}, {center_y}) → ({safe_x}, {safe_y})")
        
        return (safe_x, safe_y)    

    def find_intersection_center_independent(self, skeleton, junctions):
        """Find intersection center using ONLY skeleton and junction data, not self.roads"""
        print("  🎯 Finding intersection center independently...")
        
        # Method 1: Junction-based (most reliable)
        if junctions:
            h, w = skeleton.shape
            edge_margin = 100
            
            # Filter junctions to center region
            center_junctions = []
            for j in junctions:
                jy, jx = j
                if (edge_margin < jx < w - edge_margin and 
                    edge_margin < jy < h - edge_margin):
                    center_junctions.append(j)
            
            if center_junctions:
                junction_center_y = int(np.mean([j[0] for j in center_junctions]))
                junction_center_x = int(np.mean([j[1] for j in center_junctions]))
                center = (junction_center_x, junction_center_y)
                print(f"  ✅ Junction-based center: {center}")
                return self.validate_intersection_position(center)
        
        # Method 2: Skeleton density (fallback)
        center = self.find_intersection_center_from_skeleton(skeleton)
        print(f"  ✅ Skeleton-based center: {center}")
        return self.validate_intersection_position(center)
    
    def find_intersection_center_from_skeleton(self, skeleton):
        """Find actual intersection center from skeleton density"""
        # Create density map by convolving skeleton with a circular kernel
        kernel_size = 40
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        density_map = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
        
        # Find the point with maximum density (most skeleton pixels nearby)
        max_density_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
        center_y, center_x = max_density_idx
        
        return (center_x, center_y)

    ############# LARGE INTERSECTION ZONE CREATION #############
    def create_large_intersection_zone(self, center, roads_connecting, base_radius=80):
        """Create a large intersection zone that roads connect to"""
        
        # Calculate intersection size based on connected roads
        num_roads = len(roads_connecting)
        
        # Adaptive sizing based on road count and lengths
        if num_roads >= 6:  # Complex intersection
            zone_radius = base_radius * 1.5
        elif num_roads >= 4:  # Standard cross
            zone_radius = base_radius * 1.2  
        else:  # Simple intersection
            zone_radius = base_radius
        
        # Also consider road widths/importance
        max_road_length = max(len(road.get('points', [])) for road in roads_connecting) if roads_connecting else 100
        if max_road_length > 500:  # Major roads
            zone_radius *= 1.3
        
        return {
            'center': center,
            'radius': int(zone_radius),
            'zone_type': 'intersection_management_area',  # 🔥 Changed from 'circular'
            'intersection_shape': 'circular_zone',        # 🔥 More specific
            'purpose': 'intersection_management',         # 🔥 Clarify purpose
            'is_road': False,                            # 🔥 Explicitly NOT a road
            'is_intersection': True,                     # 🔥 Explicitly IS intersection
            'boundary_points': self.generate_intersection_boundary_points(center, zone_radius),
            'connection_points': self.generate_connection_points(center, zone_radius, len(roads_connecting)),
            'usage_explanation': 'Roads connect to this zone for navigation - this zone itself is not a driving path'  # 🔥 Usage clarification
        }



    def generate_intersection_boundary_points(self, center, radius, num_points=32):
        """Generate points around intersection boundary"""
        boundary_points = []
        center_x, center_y = center
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            boundary_points.append((int(x), int(y)))
        
        return boundary_points

    def generate_connection_points(self, center, radius, num_roads):
        """Generate specific connection points for roads to attach to"""
        connection_points = []
        center_x, center_y = center
        
        # Distribute connection points evenly around the circle
        for i in range(num_roads * 2):  # More connection points than roads for flexibility
            angle = (2 * np.pi * i) / (num_roads * 2)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            connection_points.append((int(x), int(y)))
        
        return connection_points

    def road_connects_to_point(self, road, center, max_distance=150):
        """Check if road connects to intersection center within distance"""
        points = road.get('points', [])
        if not points:
            return False
        
        center_x, center_y = center
        
        # Check start/end points
        start_dist = sqrt((points[0][0] - center_x)**2 + (points[0][1] - center_y)**2)
        end_dist = sqrt((points[-1][0] - center_x)**2 + (points[-1][1] - center_y)**2)
        
        # Also check middle points (for roads passing through)
        min_dist = min(start_dist, end_dist)
        if len(points) > 10:
            for i in range(len(points)//4, 3*len(points)//4, max(1, len(points)//20)):
                point = points[i]
                dist = sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                min_dist = min(min_dist, dist)
        
        return min_dist <= max_distance

    def find_major_intersections_with_zones(self, junctions, skeleton):
        """FIXED: Intersection detection should NOT depend on self.roads during detection phase"""
        
        print(f"🚦 CREATING LARGE INTERSECTION ZONES...")
        
        # CRITICAL FIX: Find intersection center INDEPENDENTLY of self.roads
        # Use only skeleton and junctions data
        intersection_center = self.find_intersection_center_independent(skeleton, junctions)
        
        if intersection_center is None:
            return []
        
        # AFTER finding center, then find connecting roads
        connecting_roads = []
        for road in self.roads:
            if self.road_connects_to_point(road, intersection_center, max_distance=150):
                connecting_roads.append(road)
        
        # Rest same...
        intersection_zone = self.create_large_intersection_zone(
            intersection_center, connecting_roads, base_radius=80)
        
        major_intersections = [{
            'id': 0,
            'center': intersection_center,
            'zone': intersection_zone,
            'roads_count': len(connecting_roads),
            'junction_points': [(int(x), int(y)) for y, x in junctions] if junctions else [],
            'is_zone_intersection': True,
            'detection_method': 'independent_detection'
        }]
        
        return major_intersections

    def consolidate_intersections(self, intersections, expected_intersection_count, intersection_radius=150):
        """Force consolidation when target is 1"""
        print(f"🚦 CONSOLIDATING INTERSECTIONS (targeting {expected_intersection_count})...")
        
        if len(intersections) <= expected_intersection_count:
            print(f"✅ Intersection count ({len(intersections)}) matches or is below target")
            return intersections
        
        print(f"  📊 Current: {len(intersections)} intersections, Target: {expected_intersection_count}")
        
        # BULLETPROOF: When target is 1, ALWAYS consolidate ALL intersections
        if expected_intersection_count == 1:
            print("  🔧 Target is 1 intersection - FORCE consolidating ALL intersections")
            
            if len(intersections) == 0:
                return []
            
            # Calculate centroid of all intersections
            all_centers = [i['center'] for i in intersections]
            center_x = int(np.mean([c[0] for c in all_centers]))
            center_y = int(np.mean([c[1] for c in all_centers]))
            
            # Sum all road counts
            total_roads_count = sum(i.get('roads_count', 3) for i in intersections)
            final_roads_count = min(total_roads_count, 8)
            
            # Collect all junction points
            all_junction_points = []
            for i in intersections:
                all_junction_points.extend(i.get('junction_points', []))
            
            # Get zones from original intersections
            original_zones = [i.get('zone', {}) for i in intersections if 'zone' in i]
            if original_zones:
                # Use the largest zone as base, adjust radius
                largest_zone = max(original_zones, key=lambda z: z.get('radius', 0))
                zone_radius = max(80, largest_zone.get('radius', 80))
            else:
                zone_radius = 80
            
            # Create consolidated zone
            connecting_roads = []
            for road in self.roads:
                if self.road_connects_to_point(road, (center_x, center_y), max_distance=zone_radius * 1.5):
                    connecting_roads.append(road)
            
            consolidated_zone = self.create_large_intersection_zone(
                (center_x, center_y), connecting_roads, base_radius=zone_radius)
            
            # Create single consolidated intersection
            consolidated_intersection = {
                'id': 0,
                'center': (center_x, center_y),
                'zone': consolidated_zone,
                'roads_count': final_roads_count,
                'junction_points': all_junction_points,
                'consolidated_from': [i.get('id', 0) for i in intersections],
                'cluster_size': len(intersections),
                'force_consolidated': True,
                'is_zone_intersection': True
            }
            
            print(f"  🔧 FORCE consolidated ALL {len(intersections)} intersections into single intersection at ({center_x}, {center_y})")
            print(f"  ✅ Final result: 1 intersection (was {len(intersections)})")
            
            return [consolidated_intersection]
        
        # For other targets > 1, return as-is
        return intersections

    # ==================== ROAD TRIMMING METHODS ====================
    
    def road_intersects_zone(self, road, center, radius):
        """Check if road intersects with intersection zone"""
        points = road.get('points', [])
        if not points:
            return False
        
        center_x, center_y = center
        
        # Check if any point is inside the zone
        for point in points:
            dist = sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            if dist <= radius:
                return True
        
        return False

    def find_boundary_intersection_point(self, outside_point, inside_point, center, radius):
        """Find where line crosses zone boundary"""
        from shapely.geometry import LineString, Point
        
        try:
            # Create line from outside to inside point
            line = LineString([outside_point, inside_point])
            
            # Create circle for intersection zone
            circle_center = Point(center)
            circle = circle_center.buffer(radius)
            
            # Find intersection
            intersection = line.intersection(circle.boundary)
            
            if intersection.is_empty:
                return None
            
            # Get the intersection point
            if hasattr(intersection, 'coords'):
                coords = list(intersection.coords)
                return [coords[0][0], coords[0][1]]  # First intersection point
            elif hasattr(intersection, 'x'):
                return [intersection.x, intersection.y]
            
        except Exception as e:
            print(f"      ⚠️  Boundary intersection calculation failed: {e}")
        
        return None

    def get_road_zone_entry_exit_points(self, points, center, radius):
        """Get points where road enters and exits zone"""
        center_x, center_y = center
        boundary_points = []
        
        for i in range(len(points)):
            dist = sqrt((points[i][0] - center_x)**2 + (points[i][1] - center_y)**2)
            
            # Find points near boundary
            if abs(dist - radius) < 10:  # Within 10 pixels of boundary
                boundary_points.append(points[i])
        
        return boundary_points[:2] if len(boundary_points) >= 2 else points[:2]

    def trim_road_at_zone(self, road, center, radius):
        """Trim road at intersection zone boundary, return segments"""
        points = road.get('points', [])
        if not points:
            return []
        
        center_x, center_y = center
        segments = []
        current_segment = []
        
        for i, point in enumerate(points):
            dist = sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            
            if dist > radius:
                # Point is outside zone - add to current segment
                current_segment.append(point)
            else:
                # Point is inside zone
                if current_segment:
                    # We have a segment ending at zone boundary
                    # Add boundary intersection point
                    boundary_point = self.find_boundary_intersection_point(
                        current_segment[-1], point, center, radius)
                    if boundary_point:
                        current_segment.append(boundary_point)
                    
                    # Save this segment if it's long enough
                    if len(current_segment) >= 10:  # Minimum segment length
                        segments.append({
                            'points': current_segment.copy(),
                            'segment_type': 'approaches_intersection',
                            'intersection_connection_point': boundary_point
                        })
                    
                    current_segment = []
                
                # Skip points inside the zone
                continue
        
        # Handle remaining segment after zone
        if current_segment and len(current_segment) >= 10:
            segments.append({
                'points': current_segment,
                'segment_type': 'leaves_intersection'  
            })
        
        # If no segments (road entirely in zone), create a minimal segment
        if not segments:
            # Create short segment from zone boundary
            boundary_points = self.get_road_zone_entry_exit_points(points, center, radius)
            if boundary_points:
                segments.append({
                    'points': boundary_points,
                    'segment_type': 'zone_boundary_segment'
                })
        
        return segments

    def trim_roads_at_intersection_zones(self):
        """CRITICAL: Cut roads at intersection boundaries to create clean connections"""
        print("✂️  TRIMMING ROADS AT INTERSECTION BOUNDARIES...")
        
        for intersection in self.intersections:
            int_id = intersection['id']
            center = intersection['center']
            zone = intersection.get('zone', {})
            zone_radius = zone.get('radius', 80)
            
            print(f"  ✂️  Processing intersection {int_id} (radius: {zone_radius})")
            
            # Find roads that need trimming
            roads_to_trim = []
            for road in self.roads:
                if self.road_intersects_zone(road, center, zone_radius):
                    roads_to_trim.append(road)
            
            print(f"    📍 Found {len(roads_to_trim)} roads to trim")
            
            # Trim each road
            for road in roads_to_trim:
                original_length = len(road['points'])
                trimmed_segments = self.trim_road_at_zone(road, center, zone_radius)
                
                # Replace the original road with trimmed segment(s)
                road_index = self.roads.index(road)
                self.roads.remove(road)
                
                # Insert trimmed segments with COMPLETE metadata
                for i, segment in enumerate(trimmed_segments):
                    segment['id'] = f"{road['id']}_{i}" if len(trimmed_segments) > 1 else road['id']
                    segment['original_road_id'] = road['id']
                    segment['trimmed_from_intersection'] = int_id
                    
                    # COPY ALL METADATA from original road
                    segment['metadata'] = road.get('metadata', {}).copy()
                    
                    # Ensure basic metadata structure exists
                    if 'metadata' not in segment:
                        segment['metadata'] = {}
                    
                    self.roads.insert(road_index + i, segment)
                
                trimmed_length = sum(len(seg['points']) for seg in trimmed_segments)
                print(f"      ✂️  Road {road['id']}: {original_length} → {trimmed_length} points ({len(trimmed_segments)} segments)")

    # ==================== COMPREHENSIVE METADATA METHODS ====================
    
    def create_fallback_narrative_metadata(self):
        """Create basic narrative metadata when LLM analysis fails"""
        road_narratives = []
        for i, road in enumerate(self.roads):
            road_narratives.append({
                "road_index": i,
                "road_id": road['id'],
                "conversational_identifiers": [f"Road {i}", f"the road number {i}"],
                "user_likely_descriptions": [f"road {i}", f"that road"],
                "visual_characteristics": {
                    "width_description": "standard road",
                    "surface_appearance": "paved",
                    "lane_count_apparent": "2 lanes",
                    "condition": "normal"
                },
                "directional_narrative": {
                    "where_it_comes_from": "unknown area",
                    "where_it_goes_to": "unknown area", 
                    "user_direction_language": ["that way", "this direction"],
                    "opposite_direction_language": ["the other way", "back"]
                },
                "landmark_references": [],
                "route_context": {
                    "entry_points": ["unknown"],
                    "exit_points": ["unknown"],
                    "common_destinations": ["unknown"]
                }
            })
        
        intersection_narratives = []
        for i, intersection in enumerate(self.intersections):
            intersection_narratives.append({
                "intersection_index": i,
                "intersection_id": intersection['id'],
                "conversational_identifiers": [f"Intersection {i}", "the crossing"],
                "user_likely_descriptions": [f"intersection {i}", "that crossing"],
                "directional_references": {
                    "from_here_you_can": ["go different directions"],
                    "common_navigation_language": ["at the intersection"]
                },
                "landmark_context": [],
                "route_decision_point": {
                    "why_users_mention_it": "navigation decision point",
                    "typical_user_context": ["at the intersection"]
                }
            })
        
        return {
            "road_narratives": road_narratives,
            "intersection_narratives": intersection_narratives,
            "area_context": {
                "north_area": {"user_descriptions": ["north", "up"], "apparent_land_use": "unknown", "navigation_context": "northward"},
                "south_area": {"user_descriptions": ["south", "down"], "apparent_land_use": "unknown", "navigation_context": "southward"},
                "east_area": {"user_descriptions": ["east", "right"], "apparent_land_use": "unknown", "navigation_context": "eastward"},
                "west_area": {"user_descriptions": ["west", "left"], "apparent_land_use": "unknown", "navigation_context": "westward"}
            },
            "common_route_patterns": []
        }

    def generate_narrative_metadata_for_route_matching(self, roadmap_path, satellite_path):
        """Generate narrative metadata tied to specific roads for route matching"""
        print("💬 GENERATING NARRATIVE METADATA FOR ROUTE MATCHING...")
        
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        roadmap_b64 = self.encode_image_to_base64(roadmap_path)
        satellite_b64 = self.encode_image_to_base64(satellite_path)
        
        # First, analyze the current road structure
        road_summary = []
        for i, road in enumerate(self.roads):
            points = road['points']
            if not points:
                continue
                
            start = points[0]
            end = points[-1]
            
            # Get image position context
            h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
            start_pos = self.point_to_image_position(start, w, h)
            end_pos = self.point_to_image_position(end, w, h)
            
            road_summary.append({
                "road_index": i,
                "road_id": road['id'],
                "start_position": start_pos,
                "end_position": end_pos,
                "flow_description": f"from {start_pos} to {end_pos}",
                "length_category": "long" if len(points) > 100 else "medium" if len(points) > 50 else "short"
            })
        
        intersection_summary = []
        for i, intersection in enumerate(self.intersections):
            center = intersection['center']
            center_pos = self.point_to_image_position(center, w, h)
            roads_count = intersection.get('roads_count', 4)
            
            intersection_summary.append({
                "intersection_index": i,
                "intersection_id": intersection['id'],
                "center_position": center_pos,
                "roads_connecting": roads_count,
                "zone_radius": intersection.get('zone', {}).get('radius', 80)
            })

        prompt = f"""
        TASK: Create NARRATIVE METADATA for each specific road and intersection for route matching with user conversations.

        You are analyzing a road network where users will describe their routes in natural language. 
        Create rich narrative descriptions for EACH SPECIFIC ROAD and INTERSECTION that users might reference.

        CURRENT ROAD NETWORK STRUCTURE:
        Roads: {json.dumps(road_summary, indent=2)}
        Intersections: {json.dumps(intersection_summary, indent=2)}

        Look at the MAP IMAGES and create descriptions that help match user language to specific roads/intersections.

        Roads are BIDIRECTIONAL, so consider both directions expressed precisely in conversational_identifiers and user_likely_descriptions. Like "I'm from northwest to the intersection" and "I'm heading to northwest from the intersection" is mentioning the same road.

        Return EXACTLY this JSON structure:

        {{
        "road_narratives": [
            {{
            "road_index": 0,
            "road_id": "road_0_0",
            "conversational_identifiers": [
                "the main road going northwest/upperleft from the collision point(intersection)",
                "the main road going from northwest/upperleft to the collision point(intersection)",
                "the wide street toward the northwest/upperleft from the collision point(intersection)",
                "the wide street from the northwest/upperleft to the collision point(intersection)"
            ],
            "user_likely_descriptions": [
                "I came down the main road from northwest/upperleft to the collision point(intersection)",
                "I was heading up the main road toward northwest/upperleft from the collision point(intersection)",
                "that wide street going northwest/upperleft from the collision point(intersection)",
                "that wide street coming from northwest/upperleft to the collision point(intersection)"
            ],
            "visual_characteristics": {{
                "width_description": "wide main road",
                "surface_appearance": "paved street",
                "lane_count_apparent": "appears to be 2-4 lanes",
                "condition": "well-maintained"
            }},
            "directional_narrative": {{
                "where_it_comes_from": "from the southern area", 
                "where_it_goes_to": "toward the northern area",
                "user_direction_language": ["going up", "heading north", "toward the top"],
                "opposite_direction_language": ["coming down", "from the north", "from up there"]
            }},
            "landmark_references": [
                {{
                "landmark_type": "building",
                "description": "passes by the central building",
                "user_might_say": ["by the big building", "near that large structure", "past the main building"]
                }}
            ],
            "route_context": {{
                "entry_points": ["comes from the southern area", "enters from bottom"],
                "exit_points": ["leads to the northern area", "goes toward the top"],
                "common_destinations": ["residential area", "northern district"]
            }}
            }}
        ],
        "intersection_narratives": [
            {{
            "intersection_index": 0,
            "intersection_id": 0,
            "conversational_identifiers": [
                "the main intersection",
                "the central crossing",
                "where all the roads meet"
            ],
            "user_likely_descriptions": [
                "the intersection in the middle",
                "that big crossing",
                "where the roads cross"
            ],
            "directional_references": {{
                "from_here_you_can": [
                "go straight to continue north",
                "turn left to head west", 
                "turn right to go east",
                "turn around to go back south"
                ],
                "common_navigation_language": [
                "at the intersection, turn left",
                "when you reach the crossing, go straight",
                "at the main intersection, make a right"
                ]
            }},
            "landmark_context": [
                {{
                "landmark_description": "located near the central building",
                "user_references": ["by the big building", "near that main structure"]
                }}
            ],
            "route_decision_point": {{
                "why_users_mention_it": "main decision point for navigation",
                "typical_user_context": ["I turned left at the intersection", "I went straight through the crossing"]
            }}
            }}
        ],
        "area_context": {{
            "north_area": {{
            "user_descriptions": ["up north", "toward the top", "the upper area", "going up there"],
            "apparent_land_use": "appears residential",
            "navigation_context": "where people go when heading north"
            }},
            "south_area": {{
            "user_descriptions": ["down south", "toward the bottom", "the lower area", "going down"],
            "apparent_land_use": "appears commercial", 
            "navigation_context": "where people come from when entering from south"
            }},
            "east_area": {{
            "user_descriptions": ["to the right", "eastward", "toward the right side"],
            "apparent_land_use": "mixed development",
            "navigation_context": "accessible via right turns"
            }},
            "west_area": {{
            "user_descriptions": ["to the left", "westward", "toward the left side"],
            "apparent_land_use": "mixed development",
            "navigation_context": "accessible via left turns"
            }}
        }},
        "common_route_patterns": [
            {{
            "pattern_description": "north-south main route",
            "roads_involved": [0, 1],
            "user_might_describe_as": ["I took the main road going up", "I came down the main street"]
            }}
        ]
        }}

        IMPORTANT: 
        - Match each road_index and intersection_index to the numbers provided
        - Focus on natural language users would actually use
        - Consider how people give directions and describe routes
        - Think about landmarks people actually notice and mention
        """
        
        try:
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 6000,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": roadmap_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": satellite_b64}}
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            print(f"  ✅ Received narrative metadata analysis: {len(analysis_text)} characters")
            
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                try:
                    narrative_metadata = json.loads(json_text)
                    
                    self.narrative_metadata = narrative_metadata
                    
                    road_narratives = narrative_metadata.get('road_narratives', [])
                    intersection_narratives = narrative_metadata.get('intersection_narratives', [])
                    
                    print(f"  📋 Generated narratives for {len(road_narratives)} roads")
                    print(f"  🚦 Generated narratives for {len(intersection_narratives)} intersections")
                    
                    return narrative_metadata
                    
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parsing failed: {e}")
                    print(f"  📄 Raw response preview: {analysis_text[:500]}...")
                    return self.create_fallback_narrative_metadata()
            
        except Exception as e:
            print(f"  💥 Narrative metadata generation failed: {e}")
        
        return self.create_fallback_narrative_metadata()

    def create_position_based_context(self, metadata):
        """Create directional context based on image positions rather than compass"""
        layout_desc = metadata.get('image_layout_description', '')
        
        return {
            "top_area": "leads off map northward",
            "bottom_area": "leads off map southward", 
            "left_area": "leads off map westward",
            "right_area": "leads off map eastward",
            "layout_pattern": layout_desc
        }

    def map_road_count_to_size(self, road_count):
        """Map road count to intersection size"""
        if road_count >= 6:
            return 'large'
        elif road_count >= 4:
            return 'medium'
        else:
            return 'small'

    def classify_road_from_width(self, width):
        """Map visual width to road classification"""
        width_map = {
            'very_wide': 'major_highway',
            'wide': 'main_street', 
            'medium': 'secondary_road',
            'narrow': 'local_street'
        }
        return width_map.get(width, 'secondary_road')
        
    def extract_direction_from_visual_description(self, description):
        """Extract direction from visual description"""
        desc_lower = description.lower()
        
        # Map visual descriptions to direction patterns
        if 'top_left to bottom_right' in desc_lower or 'bottom_right to top_left' in desc_lower:
            return 'diagonal_ne_sw'
        elif 'top_right to bottom_left' in desc_lower or 'bottom_left to top_right' in desc_lower:
            return 'diagonal_nw_se'
        elif any(phrase in desc_lower for phrase in ['top to bottom', 'vertical', 'north_south']):
            return 'vertical'
        elif any(phrase in desc_lower for phrase in ['left to right', 'horizontal', 'east_west']):
            return 'horizontal'
        elif 'center' in desc_lower:
            return 'center_area'
        else:
            return 'unknown_direction'   
    
    def transform_new_metadata_format(self, new_metadata):
        """Transform new specific metadata format to existing code expectations"""
        
        # Extract visible text labels
        visible_texts = new_metadata.get('visible_text_labels', [])
        road_labels = []
        landmarks = []
        
        for text_item in visible_texts:
            text = text_item.get('text', '')
            position = text_item.get('position', 'center')
            confidence = text_item.get('confidence', 'medium')
            
            # Skip low confidence or empty items
            if confidence == 'low' or not text or '?' in text:
                continue
                
            # Determine if it's a road name or landmark
            if any(keyword in text.lower() for keyword in ['route', 'road', 'street', 'ave', 'blvd', 'highway', 'hwy']):
                road_labels.append({
                    "text": text,
                    "position": position,
                    "type": "road_name",
                    "confidence": confidence
                })
            else:
                landmarks.append({
                    "name": text,
                    "type": "building",
                    "position": position,
                    "confidence": confidence
                })
        
        # Add visible landmarks
        visible_landmarks = new_metadata.get('visible_landmarks', [])
        for landmark in visible_landmarks:
            if landmark.get('name') and '?' not in landmark.get('name', ''):
                landmarks.append({
                    "name": landmark.get('name', ''),
                    "type": landmark.get('type', 'building'),
                    "position": landmark.get('position', 'center'),
                    "description": landmark.get('description', '')
                })
        
        # Transform road analysis
        road_visual_analysis = new_metadata.get('road_visual_analysis', [])
        road_classifications = []
        
        for i, road_analysis in enumerate(road_visual_analysis):
            visual_desc = road_analysis.get('visual_description', '')
            
            # Extract direction from visual description
            road_direction = self.extract_direction_from_visual_description(visual_desc)
            
            road_classifications.append({
                "road_direction": road_direction,
                "visual_description": visual_desc,
                "classification": self.classify_road_from_width(road_analysis.get('width', 'medium')),
                "width": road_analysis.get('width', 'medium'),
                "lanes": str(road_analysis.get('estimated_lanes', 2)),
                "surface": road_analysis.get('surface_appears', 'paved')
            })
        
        # Create directional context based on image positions
        directional_context = self.create_position_based_context(new_metadata)
        
        # Intersection characteristics
        intersection_analysis = new_metadata.get('intersection_analysis', {})
        intersection_characteristics = {
            "type": intersection_analysis.get('type', 'cross_intersection'),
            "size": self.map_road_count_to_size(intersection_analysis.get('estimated_road_count', 4)),
            "traffic_volume": 'high' if intersection_analysis.get('appears_signalized', False) else 'medium',
            "has_crosswalks": intersection_analysis.get('appears_signalized', False),
            "location": intersection_analysis.get('location', 'center')
        }
        
        return {
            "road_labels": road_labels,
            "road_classifications": road_classifications,
            "landmarks": landmarks,
            "directional_context": directional_context,
            "intersection_characteristics": intersection_characteristics,
            "raw_visual_analysis": new_metadata  # Keep original for debugging
        }
        
    def extract_comprehensive_metadata_with_bedrock(self, roadmap_path, satellite_path):
        """FIXED: Extract specific, actionable metadata from map images"""
        print("🏷️  EXTRACTING SPECIFIC ROAD METADATA FROM MAP IMAGES...")
        
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        roadmap_b64 = self.encode_image_to_base64(roadmap_path)
        satellite_b64 = self.encode_image_to_base64(satellite_path)
        
        prompt = """
        TASK: Extract SPECIFIC, VISIBLE information from these map images. Do NOT make up or guess information.

        FOCUS ON:
        1. EXACT TEXT VISIBLE on the maps (road names, route numbers, station names, building names)
        2. SPECIFIC LOCATIONS using image positions (not compass directions)
        3. CLEAR VISUAL FEATURES you can actually see

        FOR POSITIONS, use these 9 zones based on image layout:
        - "top_left", "top_center", "top_right" 
        - "center_left", "center", "center_right"
        - "bottom_left", "bottom_center", "bottom_right"

        IMPORTANT RULES:
        - If you can't read a text clearly, don't include it
        - Use ONLY what you can actually see in the images
        - For road directions, describe based on visual flow (e.g., "runs from top_left to bottom_right")
        - Don't guess road classifications - only use what's visually obvious

        Return ONLY this JSON format:
        {
        "visible_text_labels": [
            {"text": "Route 50", "position": "center", "confidence": "high"},
            {"text": "Main Station", "position": "top_center", "confidence": "medium"}
        ],
        "visible_landmarks": [
            {"name": "Station Building", "type": "transit", "position": "center", "description": "Large rectangular building with platform"},
            {"name": "Shopping Area", "type": "commercial", "position": "bottom_right", "description": "Cluster of small buildings"}
        ],
        "road_visual_analysis": [
            {"visual_description": "Wide road running from top_left to bottom_right", "width": "wide", "estimated_lanes": "4", "surface_appears": "paved"},
            {"visual_description": "Narrow road in center area", "width": "narrow", "estimated_lanes": "2", "surface_appears": "paved"}
        ],
        "intersection_analysis": {
            "location": "center",
            "type": "multiple roads meeting",
            "appears_signalized": true,
            "estimated_road_count": 4
        },
        "image_layout_description": "Cross-pattern intersection with roads extending to corners of image"
        }
        """
        
        try:
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "temperature": 0.1,  # Lower temperature for more factual responses
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": roadmap_b64}},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": satellite_b64}}
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            print(f"  ✅ Received specific metadata analysis: {len(analysis_text)} characters")
            
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                try:
                    metadata = json.loads(json_text)
                    
                    # Transform to compatible format for existing code
                    transformed_metadata = self.transform_new_metadata_format(metadata)
                    self.comprehensive_metadata = transformed_metadata
                    
                    print(f"  📋 Found {len(transformed_metadata.get('road_labels', []))} specific text labels")
                    print(f"  🏢 Found {len(transformed_metadata.get('landmarks', []))} visible landmarks")
                    print(f"  🛣️  Analyzed {len(transformed_metadata.get('road_classifications', []))} visible roads")
                    
                    return transformed_metadata
                    
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parsing failed: {e}")
                    print(f"  📄 Raw response preview: {analysis_text[:200]}...")
                    return self.create_fallback_metadata()
            
        except Exception as e:
            print(f"  💥 Metadata extraction failed: {e}")
        
        return self.create_fallback_metadata()

    def create_fallback_metadata(self):
        """Create basic metadata when LLM analysis fails"""
        return {
            "road_labels": [],
            "road_classifications": [
                {"road_direction": "unknown", "classification": "main_street", "width": "medium", "lanes": "2", "surface": "paved"}
            ],
            "landmarks": [],
            "directional_context": {
                "north": "unknown area",
                "south": "unknown area", 
                "east": "unknown area",
                "west": "unknown area"
            },
            "intersection_characteristics": {
                "type": "standard_cross",
                "size": "medium",
                "traffic_volume": "medium",
                "has_crosswalks": False
            }
        }

    def point_to_image_position(self, point, w, h):
        """Convert point coordinates to image position (top_left, center, etc.)"""
        x, y = point
        
        # Divide image into 3x3 grid
        col = 'left' if x < w/3 else 'center' if x < 2*w/3 else 'right'
        row = 'top' if y < h/3 else 'center' if y < 2*h/3 else 'bottom'
        
        if row == 'center' and col == 'center':
            return 'center'
        else:
            return f"{row}_{col}"

    def classify_road_direction(self, road_points):
        """IMPROVED: Classify road direction using image position terminology"""
        if len(road_points) < 2:
            return {'position_flow': 'unknown', 'simple_direction': 'stationary', 'angle': 0}
        
        start = road_points[0]
        end = road_points[-1]
        
        dx = end[0] - start[0]  
        dy = end[1] - start[1]
        
        # Calculate angle
        angle = np.arctan2(-dy, dx) * 180 / np.pi
        if angle < 0:
            angle += 360
        
        # Map to image positions instead of compass directions
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        
        # Determine start and end positions
        start_pos = self.point_to_image_position(start, w, h)
        end_pos = self.point_to_image_position(end, w, h)
        
        position_flow = f"{start_pos}_to_{end_pos}"
        
        # Simplified direction
        if abs(dx) > abs(dy):
            simple_direction = 'horizontal_flow'
        else:
            simple_direction = 'vertical_flow'
        
        return {
            'position_flow': position_flow,
            'simple_direction': simple_direction,
            'angle': angle,
            'start_area': start_pos,
            'end_area': end_pos
        }

    def find_matching_classification(self, road_direction, classifications):
        """FIXED: Find best matching road classification using new direction format"""
        if not classifications:
            return {'classification': 'local_street', 'width': 'medium', 'lanes': '2', 'surface': 'paved'}
        
        # NEW: Use the updated road_direction keys
        position_flow = road_direction.get('position_flow', 'unknown')
        simple_direction = road_direction.get('simple_direction', 'unknown')
        start_area = road_direction.get('start_area', 'unknown')
        end_area = road_direction.get('end_area', 'unknown')
        
        # Try to match by direction characteristics
        for classification in classifications:
            class_direction = classification.get('road_direction', '').lower()
            visual_desc = classification.get('visual_description', '').lower()
            
            # Match by position flow
            if position_flow != 'unknown' and position_flow in class_direction:
                return classification
                
            # Match by simple direction  
            if simple_direction in class_direction:
                return classification
                
            # Match by visual description patterns
            if visual_desc:
                # Check if visual description matches our road's characteristics
                if ('horizontal' in visual_desc and 'horizontal' in simple_direction) or \
                ('vertical' in visual_desc and 'vertical' in simple_direction) or \
                ('diagonal' in visual_desc and 'diagonal' in position_flow):
                    return classification
                    
                # Match by areas mentioned in visual description
                if any(area in visual_desc for area in [start_area, end_area] if area != 'unknown'):
                    return classification
        
        # Fallback: Return first classification
        return classifications[0] if classifications else {
            'classification': 'local_street', 
            'width': 'medium', 
            'lanes': '2', 
            'surface': 'paved'
        }

    def find_nearby_labels(self, road_points, road_labels):
        """Find labels near this road"""
        if not road_points or not road_labels:
            return []
        
        nearby_labels = []
        road_center = self.get_road_center_point(road_points)
        
        for label in road_labels:
            # Determine label position relative to image
            label_position = self.interpret_label_position(label.get('position', 'center'))
            
            # Calculate distance from road center to label position
            distance = self.calculate_distance_to_position(road_center, label_position)
            
            # Consider labels within reasonable distance
            if distance < 200:  # pixels
                nearby_labels.append({
                    'text': label.get('text', ''),
                    'type': label.get('type', 'unknown'),
                    'distance': distance,
                    'position': label.get('position', 'center')
                })
        
        # Sort by distance, return closest labels first
        return sorted(nearby_labels, key=lambda x: x['distance'])[:3]  # Max 3 labels

    def find_nearby_landmarks(self, road_points, landmarks):
        """Find landmarks near this road"""
        if not road_points or not landmarks:
            return []
        
        nearby_landmarks = []
        road_center = self.get_road_center_point(road_points)
        
        for landmark in landmarks:
            # Determine landmark position
            landmark_position = self.interpret_label_position(landmark.get('position', 'center'))
            
            # Calculate distance
            distance = self.calculate_distance_to_position(road_center, landmark_position)
            
            # Consider landmarks within reasonable distance
            if distance < 150:  # pixels
                nearby_landmarks.append({
                    'name': landmark.get('name', ''),
                    'type': landmark.get('type', 'unknown'),
                    'distance': distance,
                    'position': landmark.get('position', 'center'),
                    'side': landmark.get('side', 'unknown')
                })
        
        return sorted(nearby_landmarks, key=lambda x: x['distance'])[:5]  # Max 5 landmarks

    def get_road_center_point(self, road_points):
        """Get center point of road"""
        if not road_points:
            return (CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2)
        
        center_idx = len(road_points) // 2
        return road_points[center_idx]

    def interpret_label_position(self, position_str):
        """Convert position string to coordinates"""
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        
        position_map = {
            'center': (w//2, h//2),
            'north': (w//2, h//4),
            'south': (w//2, 3*h//4),
            'east': (3*w//4, h//2),
            'west': (w//4, h//2),
            'northeast': (3*w//4, h//4),
            'northwest': (w//4, h//4),
            'southeast': (3*w//4, 3*h//4),
            'southwest': (w//4, 3*h//4),
            'north_south': (w//2, h//2),  # Center for directional roads
            'east_west': (w//2, h//2)
        }
        
        return position_map.get(position_str.lower(), (w//2, h//2))

    def calculate_distance_to_position(self, point, position):
        """Calculate distance between point and position"""
        return sqrt((point[0] - position[0])**2 + (point[1] - position[1])**2)

    def generate_road_name(self, nearby_labels, road_direction, road_index):
        """Generate appropriate road name"""
        # Use specific label if found
        for label in nearby_labels:
            if label.get('type') in ['street_name', 'route_number']:
                return label['text']
        
        # Generate descriptive name based on direction
        direction = road_direction['simple']
        return f"Road_{road_index}_{direction}"

    def road_class_to_name(self, road_class):
        """Convert road class to readable name"""
        return {
            'major_highway': 'Highway',
            'main_street': 'Street',
            'secondary_road': 'Road', 
            'local_street': 'Lane'
        }.get(road_class, 'Road')

    def generate_display_name(self, nearby_labels, road_classification):
        """IMPROVED: Generate names from actual visible labels"""
        # First priority: Use actual visible text with high confidence
        for label in nearby_labels:
            if (label.get('confidence', 'medium') in ['high', 'medium'] and 
                label.get('text') and '?' not in label.get('text', '')):
                return label['text']
        
        # Second priority: Use road classification with position info
        position_flow = road_classification.get('position_flow', 'unknown')
        road_class = road_classification.get('classification', 'local_street')
        
        # Generate descriptive name based on visual flow
        if 'center' in position_flow:
            return f"Central {self.road_class_to_name(road_class)}"
        elif 'diagonal' in position_flow:
            return f"Diagonal {self.road_class_to_name(road_class)}"
        elif 'horizontal' in position_flow:
            return f"Horizontal {self.road_class_to_name(road_class)}"
        elif 'vertical' in position_flow:
            return f"Vertical {self.road_class_to_name(road_class)}"
        else:
            return f"Local {self.road_class_to_name(road_class)}"

    def determine_road_destination(self, road_points, directional_context):
        """Determine what area this road leads to"""
        if not road_points:
            return {'start': 'unknown', 'end': 'unknown'}
        
        start_point = road_points[0]
        end_point = road_points[-1]
        
        # Determine which edges the road approaches
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        
        start_destination = "unknown"
        end_destination = "unknown"
        
        # Check start point
        if start_point[1] < h * 0.2:  # Near top
            start_destination = directional_context.get('north', 'northern area')
        elif start_point[1] > h * 0.8:  # Near bottom
            start_destination = directional_context.get('south', 'southern area')
        elif start_point[0] < w * 0.2:  # Near left
            start_destination = directional_context.get('west', 'western area')
        elif start_point[0] > w * 0.8:  # Near right
            start_destination = directional_context.get('east', 'eastern area')
        
        # Check end point
        if end_point[1] < h * 0.2:  # Near top
            end_destination = directional_context.get('north', 'northern area')
        elif end_point[1] > h * 0.8:  # Near bottom
            end_destination = directional_context.get('south', 'southern area')
        elif end_point[0] < w * 0.2:  # Near left
            end_destination = directional_context.get('west', 'western area')
        elif end_point[0] > w * 0.8:  # Near right
            end_destination = directional_context.get('east', 'eastern area')
        
        return {
            'start': start_destination,
            'end': end_destination,
            'description': f"connects {start_destination} to {end_destination}"
        }

    def estimate_speed_limit(self, road_classification):
        """Estimate speed limit based on road class"""
        speed_map = {
            'major_highway': 65,
            'main_street': 45,
            'secondary_road': 35,
            'local_street': 25
        }
        return speed_map.get(road_classification.get('classification'), 30)

    def calculate_road_priority(self, road_classification):
        """Calculate navigation priority"""
        priority_map = {
            'major_highway': 1,
            'main_street': 2, 
            'secondary_road': 3,
            'local_street': 4
        }
        return priority_map.get(road_classification.get('classification'), 3)

    def analyze_road_curvature(self, road_points):
        """Analyze road curvature"""
        if len(road_points) < 10:
            return 'straight'
        
        # Sample points along the road
        sample_indices = [0, len(road_points)//4, len(road_points)//2, 3*len(road_points)//4, -1]
        sample_points = [road_points[i] for i in sample_indices]
        
        # Calculate angle changes
        angle_changes = []
        for i in range(len(sample_points) - 2):
            p1, p2, p3 = sample_points[i], sample_points[i+1], sample_points[i+2]
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = sqrt(v1[0]**2 + v1[1]**2)
            mag2 = sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = abs(np.arccos(cos_angle) * 180 / np.pi)
                angle_changes.append(angle)
        
        if not angle_changes:
            return 'straight'
        
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        
        if avg_angle_change < 10:
            return 'straight'
        elif avg_angle_change < 30:
            return 'slight_curve'
        elif avg_angle_change < 60:
            return 'curved'
        else:
            return 'winding'

    def generate_road_name_new_format(self, nearby_labels, road_direction, road_index):
        """Generate appropriate road name using new direction format"""
        # Use specific label if found
        for label in nearby_labels:
            if (label.get('type') in ['street_name', 'route_number', 'road_name'] and
                label.get('text') and '?' not in label.get('text', '')):
                return label['text']
        
        # Generate descriptive name based on position flow
        position_flow = road_direction.get('position_flow', 'unknown')
        simple_direction = road_direction.get('simple_direction', 'unknown')
        
        if 'center' in position_flow:
            return f"Road_{road_index}_central"
        elif 'diagonal' in position_flow:
            return f"Road_{road_index}_diagonal"
        elif simple_direction == 'horizontal_flow':
            return f"Road_{road_index}_horizontal"
        elif simple_direction == 'vertical_flow':
            return f"Road_{road_index}_vertical"
        else:
            return f"Road_{road_index}_{simple_direction}"

    def determine_road_destination_new_format(self, road_points, road_direction, directional_context):
        """Determine road destination using new position-based system"""
        if not road_points:
            return {'start': 'unknown', 'end': 'unknown'}
        
        start_area = road_direction.get('start_area', 'unknown')
        end_area = road_direction.get('end_area', 'unknown')
        
        # Map areas to directional context
        area_to_context = {
            'top_left': directional_context.get('top_area', 'upper area'),
            'top_center': directional_context.get('top_area', 'upper area'),
            'top_right': directional_context.get('top_area', 'upper area'),
            'center_left': directional_context.get('left_area', 'western area'),
            'center': 'central area',
            'center_right': directional_context.get('right_area', 'eastern area'),
            'bottom_left': directional_context.get('bottom_area', 'lower area'),
            'bottom_center': directional_context.get('bottom_area', 'lower area'),
            'bottom_right': directional_context.get('bottom_area', 'lower area')
        }
        
        start_destination = area_to_context.get(start_area, f"{start_area} area")
        end_destination = area_to_context.get(end_area, f"{end_area} area")
        
        return {
            'start': start_destination,
            'end': end_destination,
            'description': f"connects {start_destination} to {end_destination}",
            'position_flow': road_direction.get('position_flow', 'unknown')
        }

    def get_edge_sides_new_format(self, road_points):
        """Get which edges the road touches using new position format"""
        if not road_points:
            return []
        
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        edge_threshold = 50
        edges = []
        
        for point in [road_points[0], road_points[-1]]:
            # Convert to position format
            position = self.point_to_image_position(point, w, h)
            
            if 'top' in position:
                edges.append('top_edge')
            elif 'bottom' in position:
                edges.append('bottom_edge')
            
            if 'left' in position:
                edges.append('left_edge')
            elif 'right' in position:
                edges.append('right_edge')
        
        return list(set(edges))  # Remove duplicates

    def assign_metadata_to_roads(self):
        """FIXED: Assign comprehensive metadata using new direction format"""
        print("🏷️  ASSIGNING METADATA TO TRIMMED ROADS...")
        
        metadata = getattr(self, 'comprehensive_metadata', {})
        road_classifications = metadata.get('road_classifications', [])
        road_labels = metadata.get('road_labels', [])
        landmarks = metadata.get('landmarks', [])
        directional_context = metadata.get('directional_context', {})
        
        for i, road in enumerate(self.roads):
            road_points = road['points']
            if not road_points:
                continue
            
            # FIXED: Use new direction classification method
            road_direction = self.classify_road_direction(road_points)
            
            # Find matching classification
            road_classification = self.find_matching_classification(road_direction, road_classifications)
            
            # Find nearby labels
            nearby_labels = self.find_nearby_labels(road_points, road_labels)
            
            # Find nearby landmarks
            nearby_landmarks = self.find_nearby_landmarks(road_points, landmarks)
            
            # Determine what this road leads to using new direction format
            leads_to = self.determine_road_destination_new_format(road_points, road_direction, directional_context)
            
            # Create comprehensive metadata
            if 'metadata' not in road:
                road['metadata'] = {}
            
            road['metadata'].update({
                # Basic identification
                'name': self.generate_road_name_new_format(nearby_labels, road_direction, i),
                'display_name': self.generate_display_name(nearby_labels, road_classification),
                'road_id': road['id'],
                
                # Classification
                'road_class': road_classification.get('classification', 'local_street'),
                'road_type': self.map_classification_to_type(road_classification.get('classification')),
                'width_category': road_classification.get('width', 'medium'),
                'estimated_lanes': int(road_classification.get('lanes', '2')),
                'surface_type': road_classification.get('surface', 'paved'),
                
                # FIXED: Use new direction format
                'position_flow': road_direction.get('position_flow', 'unknown'),
                'simple_direction': road_direction.get('simple_direction', 'unknown'),
                'start_area': road_direction.get('start_area', 'unknown'),
                'end_area': road_direction.get('end_area', 'unknown'),
                'geometric_angle': road_direction.get('angle', 0),
                
                # Context and destinations
                'leads_to': leads_to,
                'nearby_landmarks': nearby_landmarks,
                'visible_labels': nearby_labels,
                
                # Navigation metadata
                'estimated_speed_limit': self.estimate_speed_limit(road_classification),
                'traffic_density': self.estimate_traffic_density(road_classification),
                'priority': self.calculate_road_priority(road_classification),
                
                # Physical characteristics
                'curvature': self.analyze_road_curvature(road_points),
                'estimated_length_meters': len(road_points) * 0.5,
                
                # Turn restrictions (based on road class)
                'can_turn_left': road_classification.get('classification') != 'major_highway',
                'can_turn_right': True,
                'can_go_straight': True,
                'has_median': road_classification.get('classification') in ['major_highway', 'main_street'],
                'parking_available': road_classification.get('classification') in ['local_street', 'secondary_road'],
                
                # Edge analysis (updated to use new direction format)
                'edge_analysis': {
                    'has_edge_connection': self.road_touches_edge(road_points),
                    'edge_sides': self.get_edge_sides_new_format(road_points),
                    'start_position': road_direction.get('start_area', 'unknown'),
                    'end_position': road_direction.get('end_area', 'unknown')
                }
            })
            
            road_name = road['metadata']['display_name']
            road_class = road['metadata']['road_class']
            direction = road['metadata']['simple_direction']
            
            print(f"  ✅ Road {i}: {road_name} ({road_class}, {direction})")
            
    def map_classification_to_type(self, classification):
        """Map road classification to road type"""
        type_map = {
            'major_highway': 'highway',
            'main_street': 'primary',
            'secondary_road': 'secondary',
            'local_street': 'residential'
        }
        return type_map.get(classification, 'unclassified')

    def estimate_traffic_density(self, road_classification):
        """Estimate traffic density based on road class"""
        density_map = {
            'major_highway': 'high',
            'main_street': 'high',
            'secondary_road': 'medium',
            'local_street': 'low'
        }
        return density_map.get(road_classification.get('classification'), 'medium')

    def road_touches_edge(self, road_points):
        """Check if road touches image edge"""
        if not road_points:
            return False
        
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        edge_threshold = 50  # pixels from edge
        
        for point in [road_points[0], road_points[-1]]:  # Check start and end
            if (point[0] < edge_threshold or point[0] > w - edge_threshold or
                point[1] < edge_threshold or point[1] > h - edge_threshold):
                return True
        
        return False

    def get_edge_sides(self, road_points):
        """Get which edges the road touches"""
        if not road_points:
            return []
        
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        edge_threshold = 50
        edges = []
        
        for point in [road_points[0], road_points[-1]]:
            if point[1] < edge_threshold:
                edges.append('north')
            elif point[1] > h - edge_threshold:
                edges.append('south')
            elif point[0] < edge_threshold:
                edges.append('west')
            elif point[0] > w - edge_threshold:
                edges.append('east')
        
        return list(set(edges))  # Remove duplicates

    def get_true_directions(self, road_points, end_type):
        """Get true geographic directions for road endpoints"""
        if not road_points:
            return []
        
        if end_type == 'start':
            point = road_points[0]
        else:  # end
            point = road_points[-1]
        
        h, w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        directions = []
        
        # Determine directions based on position
        if point[1] < h * 0.3:
            directions.append('north')
        elif point[1] > h * 0.7:
            directions.append('south')
        
        if point[0] < w * 0.3:
            directions.append('west')
        elif point[0] > w * 0.7:
            directions.append('east')
        
        return directions

    def find_nearby_landmarks_for_point(self, point, landmarks, radius=150):
        """Find landmarks near a specific point"""
        if not landmarks:
            return []
        
        nearby = []
        for landmark in landmarks:
            landmark_position = self.interpret_label_position(landmark.get('position', 'center'))
            distance = self.calculate_distance_to_position(point, landmark_position)
            
            if distance <= radius:
                nearby.append({
                    'name': landmark.get('name', ''),
                    'type': landmark.get('type', 'unknown'),
                    'distance': distance,
                    'position': landmark.get('position', 'center')
                })
        
        return sorted(nearby, key=lambda x: x['distance'])

    def get_connected_roads_metadata(self, intersection_id):
        """Get metadata for roads connected to intersection"""
        connected_roads_info = []
        
        road_connections = getattr(self, 'road_connections', {})
        connections = road_connections.get(intersection_id, [])
        
        for connection in connections:
            road_id = connection.get('road_id')
            road = next((r for r in self.roads if r['id'] == road_id), None)
            
            if road and 'metadata' in road:
                metadata = road['metadata']
                connected_roads_info.append({
                    'road_id': road_id,
                    'display_name': metadata.get('display_name', f'Road_{road_id}'),
                    'road_class': metadata.get('road_class', 'local_street'),
                    'direction': metadata.get('simple_direction', 'unknown'),
                    'leads_to': metadata.get('leads_to', {}),
                    'priority': metadata.get('priority', 3)
                })
        
        return connected_roads_info

    def calculate_intersection_complexity(self, intersection):
        """Calculate intersection complexity score"""
        roads_count = intersection.get('roads_count', 4)
        zone_radius = intersection.get('zone', {}).get('radius', 60)
        
        if roads_count >= 6 or zone_radius > 100:
            return 'complex'
        elif roads_count >= 4 or zone_radius > 80:
            return 'medium'
        else:
            return 'simple'

    def estimate_intersection_wait_time(self, intersection_chars):
        """Estimate wait time at intersection"""
        traffic_volume = intersection_chars.get('traffic_volume', 'medium')
        has_signals = intersection_chars.get('has_crosswalks', False)
        
        base_time = 30 if has_signals else 15
        
        multiplier_map = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5
        }
        
        return int(base_time * multiplier_map.get(traffic_volume, 1.0))

    def generate_intersection_name(self, nearby_landmarks, connected_roads_info):
        """Generate descriptive intersection name"""
        # Use major landmark if available
        major_landmarks = [lm for lm in nearby_landmarks if lm.get('type') in ['transit', 'major_building']]
        if major_landmarks:
            return f"{major_landmarks[0]['name']} Intersection"
        
        # Use major roads
        major_roads = [road['display_name'] for road in connected_roads_info 
                    if road['road_class'] in ['major_highway', 'main_street']]
        if len(major_roads) >= 2:
            return f"{major_roads[0]} & {major_roads[1]}"
        
        # Use any landmarks
        if nearby_landmarks:
            return f"Near {nearby_landmarks[0]['name']}"
        
        # Generic name
        return "Main Intersection"

    def assign_narrative_metadata_to_intersections(self):
        """Assign narrative metadata to intersections for route matching"""
        print("🚦 ASSIGNING NARRATIVE METADATA TO INTERSECTIONS...")
        
        narrative_metadata = getattr(self, 'narrative_metadata', {})
        intersection_narratives = narrative_metadata.get('intersection_narratives', [])
        area_context = narrative_metadata.get('area_context', {})
        
        for i, intersection in enumerate(self.intersections):
            # Find matching narrative
            intersection_narrative = None
            for narrative in intersection_narratives:
                if (narrative.get('intersection_index') == i or 
                    narrative.get('intersection_id') == intersection['id']):
                    intersection_narrative = narrative
                    break
            
            if not intersection_narrative and intersection_narratives:
                intersection_narrative = intersection_narratives[min(i, len(intersection_narratives)-1)]
            
            if not intersection_narrative:
                intersection_narrative = {
                    "conversational_identifiers": [f"Intersection {i}"],
                    "user_likely_descriptions": [f"intersection {i}"],
                    "directional_references": {"from_here_you_can": [], "common_navigation_language": []},
                    "landmark_context": [],
                    "route_decision_point": {"why_users_mention_it": "decision point", "typical_user_context": []}
                }
            
            # Get existing metadata or create new  
            if 'metadata' not in intersection:
                intersection['metadata'] = {}
            
            # Add narrative metadata
            intersection['metadata'].update({
                # Core identification (preserve existing)
                'intersection_id': intersection['id'],
                'intersection_name': intersection['metadata'].get('intersection_name', f'Intersection_{intersection["id"]}'),
                
                # NEW: Narrative conversation data
                'conversational_identifiers': intersection_narrative.get('conversational_identifiers', []),
                'user_likely_descriptions': intersection_narrative.get('user_likely_descriptions', []),
                
                # NEW: Directional conversation context
                'navigation_conversations': intersection_narrative.get('directional_references', {}),
                
                # NEW: Landmark conversation context
                'conversation_landmarks': intersection_narrative.get('landmark_context', []),
                
                # NEW: Route decision context
                'route_decision_context': intersection_narrative.get('route_decision_point', {}),
                
                # Preserve existing technical metadata
                'intersection_type': intersection['metadata'].get('intersection_type', 'cross_intersection'),
                'traffic_volume': intersection['metadata'].get('traffic_volume', 'medium'),
                'has_traffic_signals': intersection['metadata'].get('has_traffic_signals', False)
            })
            
            # Log what we assigned
            identifiers = intersection['metadata']['conversational_identifiers'][:2]
            print(f"  ✅ Intersection {i}: {identifiers}")

    def assign_narrative_metadata_to_roads(self):
        """Assign narrative metadata to roads for route matching"""
        print("🏷️  ASSIGNING NARRATIVE METADATA TO ROADS...")
        
        narrative_metadata = getattr(self, 'narrative_metadata', {})
        road_narratives = narrative_metadata.get('road_narratives', [])
        area_context = narrative_metadata.get('area_context', {})
        route_patterns = narrative_metadata.get('common_route_patterns', [])
        
        for i, road in enumerate(self.roads):
            road_points = road['points']
            if not road_points:
                continue
            
            # Find matching narrative by road index or ID
            road_narrative = None
            for narrative in road_narratives:
                if (narrative.get('road_index') == i or 
                    narrative.get('road_id') == road['id']):
                    road_narrative = narrative
                    break
            
            # Fallback to first available or create basic one
            if not road_narrative and road_narratives:
                road_narrative = road_narratives[min(i, len(road_narratives)-1)]
            
            if not road_narrative:
                road_narrative = {
                    "conversational_identifiers": [f"Road {i}"],
                    "user_likely_descriptions": [f"road {i}"],
                    "visual_characteristics": {"width_description": "standard road"},
                    "directional_narrative": {"where_it_comes_from": "unknown", "where_it_goes_to": "unknown"},
                    "landmark_references": [],
                    "route_context": {"entry_points": [], "exit_points": [], "common_destinations": []}
                }
            
            # Get existing metadata or create new
            if 'metadata' not in road:
                road['metadata'] = {}
            
            # Add narrative metadata while preserving existing metadata
            road['metadata'].update({
                # Core identification (preserve existing)
                'road_id': road['id'],
                'name': road['metadata'].get('name', f'Road_{road["id"]}'),
                
                # NEW: Narrative conversation data
                'conversational_identifiers': road_narrative.get('conversational_identifiers', []),
                'user_likely_descriptions': road_narrative.get('user_likely_descriptions', []),
                
                # NEW: Enhanced visual description
                'narrative_visual_characteristics': road_narrative.get('visual_characteristics', {}),
                
                # NEW: Narrative directional context
                'narrative_directional': road_narrative.get('directional_narrative', {}),
                
                # NEW: Landmark context for conversations
                'conversation_landmarks': road_narrative.get('landmark_references', []),
                
                # NEW: Route context for matching user descriptions
                'route_context': road_narrative.get('route_context', {}),
                
                # Preserve existing technical metadata
                'road_class': road['metadata'].get('road_class', 'local_street'),
                'road_type': road['metadata'].get('road_type', 'street'),
                'estimated_speed_limit': road['metadata'].get('estimated_speed_limit', 30),
                'priority': road['metadata'].get('priority', 3),
                
                # Add pattern matching info
                'appears_in_route_patterns': [
                    pattern for pattern in route_patterns 
                    if i in pattern.get('roads_involved', [])
                ]
            })
            
            # Log what we assigned
            identifiers = road['metadata']['conversational_identifiers'][:2]  # First 2
            print(f"  ✅ Road {i}: {identifiers}")

    # ==================== INTERSECTION METADATA METHODS ====================
    def assign_metadata_to_intersections(self):
        """Assign comprehensive metadata to intersections - AFTER TRIMMING"""
        print("🏷️  ASSIGNING METADATA TO INTERSECTIONS...")
        
        metadata = getattr(self, 'comprehensive_metadata', {})
        landmarks = metadata.get('landmarks', [])
        intersection_chars = metadata.get('intersection_characteristics', {})
        
        for intersection in self.intersections:
            center = intersection['center']
            
            # Find nearby landmarks
            nearby_landmarks = self.find_nearby_landmarks_for_point(center, landmarks, radius=150)
            
            # Get connected roads metadata
            connected_roads_info = self.get_connected_roads_metadata(intersection['id'])
            
            if 'metadata' not in intersection:
                intersection['metadata'] = {}
            
            # 🔥 GET ZONE INFO FOR EXPLICIT INTERSECTION IDENTIFICATION
            zone = intersection.get('zone', {})
            zone_radius = zone.get('radius', 80)
            zone_type = zone.get('zone_type', 'circular')
            
            intersection['metadata'].update({
                # Basic identification
                'intersection_name': self.generate_intersection_name(nearby_landmarks, connected_roads_info),
                'intersection_id': intersection['id'],
                
                # Classification
                'intersection_type': f"{intersection.get('roads_count', 4)}-way_intersection",
                'size_category': intersection_chars.get('size', 'medium'),
                'traffic_volume': intersection_chars.get('traffic_volume', 'medium'),
                'complexity': self.calculate_intersection_complexity(intersection),
                
                # Infrastructure
                'has_traffic_signals': intersection_chars.get('has_crosswalks', False),
                'has_crosswalks': intersection_chars.get('has_crosswalks', True),
                'has_turning_lanes': any(road['road_class'] in ['major_highway', 'main_street'] 
                                    for road in connected_roads_info),
                
                # Context
                'nearby_landmarks': nearby_landmarks,
                'nearby_businesses': [lm['name'] for lm in nearby_landmarks if lm.get('type') == 'commercial'],
                
                # Navigation
                'estimated_wait_time': self.estimate_intersection_wait_time(intersection_chars),
                'navigation_complexity': self.calculate_intersection_complexity(intersection),
                
                # Connected roads summary
                'connected_roads_summary': {
                    'major_roads': [road['display_name'] for road in connected_roads_info 
                                if road['road_class'] in ['major_highway', 'main_street']],
                    'secondary_roads': [road['display_name'] for road in connected_roads_info 
                                    if road['road_class'] == 'secondary_road'],
                    'local_streets': [road['display_name'] for road in connected_roads_info 
                                    if road['road_class'] == 'local_street']
                },
                
                # 🔥 NEW: EXPLICIT INTERSECTION IDENTIFICATION TO PREVENT CIRCLE ROAD CONFUSION
                'structure_type': 'intersection_zone',
                'is_road': False,
                'is_intersection': True,
                'not_a_circular_road': True,
                'intersection_explanation': f'This is an intersection management zone with {zone_radius}px radius where {len(connected_roads_info)} roads meet. It is NOT a circular road to drive around.',
                'zone_purpose': 'traffic_management_and_turning_area',
                'driving_behavior': 'roads_connect_to_this_zone_but_do_not_circle_around_it',
                'zone_description': f'Circular management area (radius: {zone_radius}px) for intersection navigation, not a roadway',
                
                # 🔥 LLM-SPECIFIC CLARIFICATIONS
                'for_llm_analysis': {
                    'this_is_intersection_not_road': True,
                    'circular_shape_explanation': 'The circular shape is used for traffic management geometry, not as a driving path',
                    'road_relationship': 'Roads terminate at or pass through this zone - they do not circle around it',
                    'navigation_purpose': 'Intersection decision point and turning area'
                }
            })
            
            intersection_name = intersection['metadata']['intersection_name']
            nearby_count = len(nearby_landmarks)
            
            print(f"  ✅ Intersection {intersection['id']}: {intersection_name} ({nearby_count} landmarks)")
            print(f"    🔧 Added explicit metadata: NOT a circular road, IS an intersection zone")

    # ==================== LANE GENERATION METHODS ====================
    
    def create_offset_lane(self, points, offset_distance, side):
        """Create offset lane using shapely"""
        try:
            dict_points = [{"x": float(p[0]), "y": float(p[1])} for p in points]
            line = LineString([(p["x"], p["y"]) for p in dict_points])
            offset_line = line.parallel_offset(offset_distance, side=side, join_style=2)
            
            if offset_line.geom_type == "MultiLineString":
                offset_line = max(offset_line.geoms, key=lambda l: l.length)
            
            if offset_line.is_empty:
                return points
                
            offset_coords = list(offset_line.coords)
            return [[float(x), float(y)] for x, y in offset_coords]
        
        except Exception:
            return points

    def is_flipped_direction(self, points):
        """Check if line direction should be flipped"""
        if len(points) < 2:
            return False
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return (90 < angle < 270) or (-270 < angle < -90)

    def create_dual_lanes_for_road(self, road):
        """Create lanes with comprehensive metadata"""
        points = road['points']
        metadata = road.get('metadata', {})
        
        # Create offset lanes
        forward_lane = self.create_offset_lane(points, LANE_OFFSET_PX, "left")
        backward_lane = self.create_offset_lane(points, LANE_OFFSET_PX, "right")
        
        # Check if direction needs to be flipped
        if self.is_flipped_direction(points):
            forward_lane, backward_lane = backward_lane, forward_lane
        
        return {
            'forward_lane': forward_lane,
            'backward_lane': backward_lane,
            'metadata': metadata  # Pass through all metadata
        }

    def find_road_intersection_connection(self, road):
        """Find which intersection this trimmed road connects to"""
        road_id = road['id']
        
        # Check all intersections
        for intersection in self.intersections:
            int_id = intersection['id']
            zone = intersection.get('zone', {})
            center = intersection['center']
            radius = zone.get('radius', 80)
            
            # Check if road endpoint is near this intersection
            points = road['points']
            if not points:
                continue
            
            start_dist = sqrt((points[0][0] - center[0])**2 + (points[0][1] - center[1])**2)
            end_dist = sqrt((points[-1][0] - center[0])**2 + (points[-1][1] - center[1])**2)
            
            # Road connects if either end is at zone boundary
            boundary_tolerance = radius * 0.3  # 30% tolerance
            if (abs(start_dist - radius) <= boundary_tolerance or 
                abs(end_dist - radius) <= boundary_tolerance):
                
                return {
                    'intersection_id': int_id,
                    'connection_end': 'start' if start_dist < end_dist else 'end',
                    'distance_to_boundary': min(abs(start_dist - radius), abs(end_dist - radius))
                }
        
        return None

    def find_clean_branches_for_trimmed_road(self, road_id, traffic_direction, intersection_connection):
        """ENHANCED: Find branches with clean target road info and turn directions"""
        branches = []
        intersection_id = intersection_connection['intersection_id']
        
        # Find other trimmed roads connected to same intersection
        connected_roads = []
        for other_road in self.roads:
            if other_road['id'] == road_id:
                continue
            
            other_connection = self.find_road_intersection_connection(other_road)
            if (other_connection and 
                other_connection['intersection_id'] == intersection_id):
                connected_roads.append(other_road)
        
        # Create branches to connected roads
        for target_road in connected_roads:
            target_metadata = target_road.get('metadata', {})
            target_direction = target_metadata.get('simple_direction', 'unknown')
            
            # Determine turn type based on directions
            turn_type = self.calculate_turn_type(traffic_direction, target_direction)
            
            # Clean target road name
            target_road_name = target_metadata.get('display_name', f'Road_{target_road["id"]}')
            if '???' in target_road_name:
                # Fallback to clean name generation
                target_class = target_metadata.get('road_class', 'local_street')
                target_road_name = f"{target_direction} {target_class.replace('_', ' ').title()}"
            
            branch = {
                'branch_id': f"from_{road_id}_{traffic_direction}_to_{target_road['id']}",
                'target_road_id': target_road['id'],
                'target_road_name': target_road_name,
                'target_road_direction': target_direction,
                'turn_type': turn_type,
                'navigation_instruction': f"{turn_type.replace('_', ' ').title()} to {target_road_name}",
                'is_clean_connection': True,
                
                # LHT-specific info
                'lht_turn_guidance': self.get_lht_turn_guidance(turn_type, target_road_name)
            }
            
            branches.append(branch)
            print(f"    ✅ Branch: {turn_type} to {target_road_name}")
        
        return branches

    def calculate_turn_type(self, from_direction, to_direction):
        """Calculate turn type between two directions"""
        direction_angles = {"E": 0, "NE": 45, "N": 90, "NW": 135, "W": 180, "SW": 225, "S": 270, "SE": 315}
        
        from_angle = direction_angles.get(from_direction, 0)
        to_angle = direction_angles.get(to_direction, 0)
        
        # Calculate relative angle
        relative_angle = (to_angle - from_angle) % 360
        
        if 0 <= relative_angle <= 45 or 315 <= relative_angle <= 360:
            return "straight"
        elif 45 < relative_angle <= 135:
            return "left_turn"
        elif 135 < relative_angle <= 225:
            return "u_turn"
        else:
            return "right_turn"

    def get_lht_turn_guidance(self, turn_type, target_road_name):
        """Get LHT-specific turn guidance"""
        guidance_map = {
            'straight': f"Continue straight ahead to {target_road_name}",
            'left_turn': f"Turn left (across traffic) to {target_road_name}",
            'right_turn': f"Turn right (with traffic) to {target_road_name}",
            'u_turn': f"Make U-turn to {target_road_name}"
        }
        return guidance_map.get(turn_type, f"Navigate to {target_road_name}")

    def generate_clean_lane_tree(self, road, direction, intersection_connection):
        """FIXED: Generate lane with CLEAN names and GEOGRAPHIC directions"""
        road_id = road['id']
        
        # Get the CURRENT road metadata (after LLM enrichment)
        current_road = next((r for r in self.roads if r['id'] == road_id), road)
        road_metadata = current_road.get('metadata', {})
        
        # Create clean offset lanes
        dual_lanes = self.create_dual_lanes_for_road(current_road)
        
        # CRITICAL FIX 1: Convert forward/backward to geographic directions
        # Get road's actual geographic direction
        points = road['points']
        if len(points) >= 2:
            start_point = points[0]
            end_point = points[-1]
            
            # Calculate actual heading
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]  # Note: may need to flip based on coordinate system
            
            # Convert to geographic direction
            import math
            angle = math.atan2(-dy, dx) * 180 / math.pi  # -dy because image coords have y increasing downward
            if angle < 0:
                angle += 360
                
            # Map to 8-way directions
            directions_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            direction_index = int((angle + 22.5) / 45) % 8
            geographic_direction = directions_8[direction_index]
            
            # Opposite direction
            opposite_index = (direction_index + 4) % 8
            opposite_direction = directions_8[opposite_index]
            
            # Determine which lane direction to use
            if direction == 'forward':
                lane_direction = geographic_direction
                trunk_points = dual_lanes['forward_lane']
            else:  # backward
                lane_direction = opposite_direction  
                trunk_points = dual_lanes['backward_lane'][::-1]
        else:
            # Fallback
            lane_direction = direction
            trunk_points = dual_lanes['forward_lane'] if direction == 'forward' else dual_lanes['backward_lane'][::-1]
        
        # CRITICAL FIX 2: Generate CLEAN display name (NO ???)
        import copy
        complete_metadata = copy.deepcopy(road_metadata)
        
        # Generate meaningful lane name
        conv_ids = complete_metadata.get('conversational_identifiers', [])
        user_descriptions = complete_metadata.get('user_likely_descriptions', [])
        
        if conv_ids and len(conv_ids) > 0 and '???' not in conv_ids[0]:
            # Use conversational identifier
            base_name = conv_ids[0].title()
            lane_display_name = f"{base_name} ({lane_direction})"
        elif user_descriptions and len(user_descriptions) > 0 and '???' not in user_descriptions[0]:
            # Use user description
            base_name = user_descriptions[0].title()
            lane_display_name = f"{base_name} ({lane_direction})"
        else:
            # Generate from road class and geographic direction
            road_class = complete_metadata.get('road_class', 'local_street')
            
            class_names = {
                'major_highway': 'Highway',
                'main_street': 'Main Street',
                'secondary_road': 'Road',
                'local_street': 'Street'
            }
            
            class_name = class_names.get(road_class, 'Road')
            lane_display_name = f"{lane_direction} {class_name}"
        
        # Update metadata with clean information
        complete_metadata['display_name'] = lane_display_name
        complete_metadata['name'] = lane_display_name
        complete_metadata['geographic_direction'] = lane_direction
        complete_metadata['original_direction'] = direction  # Keep for reference
        
        print(f"  ✅ Lane: {lane_display_name} (was: {direction})")
        
        # FIXED lane tree with geographic direction and clean name
        lane_tree = {
            'lane_id': f"road_{road_id}_{lane_direction}_lane",  # Use geographic direction in ID
            'road_id': road_id,
            'direction': lane_direction,  # NOW GEOGRAPHIC (N/S/E/W/etc)
            'lane_type': 'trunk',
            'points': trunk_points,
            'branches': [],
            'metadata': complete_metadata  # Clean display name, no ???
        }
        
        # Add branches if connected to intersection
        if intersection_connection:
            branches = self.find_clean_branches_for_trimmed_road(
                road_id, lane_direction, intersection_connection)  # Pass geographic direction
            lane_tree['branches'] = branches
            lane_tree['intersection_connection'] = intersection_connection
        
        return lane_tree

    def generate_geographic_lanes_from_roads(self):
        """Generate lanes based on actual traffic flow, not arbitrary forward/backward"""
        print("🛤️  GENERATING TRAFFIC-FLOW BASED LANES...")
        
        self.lane_trees = []
        
        for road in self.roads:
            points = road['points']
            if len(points) < 5:
                continue
                
            road_metadata = road.get('metadata', {})
            intersection_connection = self.find_road_intersection_connection(road)
            
            # Calculate the geographic direction of this road
            start_point = points[0]
            end_point = points[-1]
            
            import math
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.atan2(-dy, dx) * 180 / math.pi
            if angle < 0:
                angle += 360
                
            directions_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            direction_index = int((angle + 22.5) / 45) % 8
            primary_direction = directions_8[direction_index]
            opposite_direction = directions_8[(direction_index + 4) % 8]
            
            # Generate lane for each traffic direction
            # In LHT (Japan), traffic flows in both directions on most roads
            lane_1 = self.generate_traffic_flow_lane(road, primary_direction, intersection_connection)
            lane_2 = self.generate_traffic_flow_lane(road, opposite_direction, intersection_connection)
            
            self.lane_trees.extend([lane_1, lane_2])
            
            print(f"    ✅ Generated {primary_direction}/{opposite_direction} traffic lanes for Road {road['id']}")

    def add_lht_lane_usage_guidance(self):
        """Add LHT-specific lane usage guidance for navigation"""
        print("🚗 ADDING LHT LANE USAGE GUIDANCE...")
        
        for lane_tree in self.lane_trees:
            metadata = lane_tree.get('metadata', {})
            traffic_direction = lane_tree.get('direction', 'unknown')
            branches = lane_tree.get('branches', [])
            
            # For LHT, determine when this lane should be used
            lane_usage_guidance = {
                'use_when_heading': traffic_direction,
                'suitable_for_turns': [],
                'lane_position': 'left' if 'left' in lane_tree.get('lane_id', '') else 'right',
                'lht_recommendations': []
            }
            
            # Analyze branches to determine turn suitability
            for branch in branches:
                turn_type = branch.get('turn_type', 'unknown')
                target_direction = branch.get('target_road_direction', 'unknown')
                
                if 'left' in turn_type.lower():
                    lane_usage_guidance['suitable_for_turns'].append('left_turn')
                    lane_usage_guidance['lht_recommendations'].append(
                        f"Use this lane when turning left to {branch.get('target_road_name', 'target road')}")
                elif 'right' in turn_type.lower():
                    lane_usage_guidance['suitable_for_turns'].append('right_turn')
                    lane_usage_guidance['lht_recommendations'].append(
                        f"Use this lane when turning right to {branch.get('target_road_name', 'target road')}")
                elif 'straight' in turn_type.lower() or 'through' in turn_type.lower():
                    lane_usage_guidance['suitable_for_turns'].append('straight')
                    lane_usage_guidance['lht_recommendations'].append(
                        f"Use this lane when going straight to {branch.get('target_road_name', 'target road')}")
            
            # Add to metadata
            metadata['lht_lane_usage'] = lane_usage_guidance
            
            print(f"  ✅ Lane {lane_tree.get('lane_id', '')}: suitable for {lane_usage_guidance['suitable_for_turns']}")

    def generate_traffic_flow_lane(self, road, traffic_direction, intersection_connection):
        """Generate lane based on actual traffic flow direction"""
        road_id = road['id']
        road_metadata = road.get('metadata', {})
        
        # Create lane geometry (this part stays similar)
        dual_lanes = self.create_dual_lanes_for_road(road)
        
        # Determine which physical lane to use based on traffic direction
        points = road['points']
        start_point = points[0]
        end_point = points[-1]
        
        # Calculate if traffic_direction matches road digitization direction
        import math
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        road_angle = math.atan2(-dy, dx) * 180 / math.pi
        if road_angle < 0:
            road_angle += 360
            
        # Convert traffic_direction to angle
        direction_angles = {"E": 0, "NE": 45, "N": 90, "NW": 135, "W": 180, "SW": 225, "S": 270, "SE": 315}
        traffic_angle = direction_angles.get(traffic_direction, 0)
        
        # If angles are similar, use "forward" lane geometry, otherwise "backward"
        angle_diff = abs(road_angle - traffic_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        if angle_diff <= 90:
            trunk_points = dual_lanes['forward_lane']
        else:
            trunk_points = dual_lanes['backward_lane'][::-1]
        
        # 🔥 FIX: Create COMPLETE lane metadata that matches road metadata structure
        lane_metadata = {
            # Basic names - create consistent display name
            'display_name': f"{road_metadata.get('display_name', f'Road_{road_id}')} ({traffic_direction} Lane)",
            'name': f"{traffic_direction} Lane of {road_metadata.get('name', f'Road_{road_id}')}",
            'traffic_direction': traffic_direction,
            'parent_road_id': road_id,
            'parent_road_name': road_metadata.get('display_name', f'Road_{road_id}'),
            
            # 🔥 FIX: PRESERVE ALL narrative metadata from parent road
            'conversational_identifiers': road_metadata.get('conversational_identifiers', []),
            'user_likely_descriptions': road_metadata.get('user_likely_descriptions', []),
            'narrative_visual_characteristics': road_metadata.get('narrative_visual_characteristics', {}),
            'narrative_directional': road_metadata.get('narrative_directional', {}),
            'conversation_landmarks': road_metadata.get('conversation_landmarks', []),
            'route_context': road_metadata.get('route_context', {}),
            'nearby_landmarks': road_metadata.get('nearby_landmarks', []),
            'visible_labels': road_metadata.get('visible_labels', []),
            'leads_to': road_metadata.get('leads_to', {}),
            
            # Copy essential road properties
            'road_class': road_metadata.get('road_class', 'local_street'),
            'road_type': road_metadata.get('road_type', 'street'),
            'width_category': road_metadata.get('width_category', 'medium'),
            'estimated_lanes': road_metadata.get('estimated_lanes', 2),
            'surface_type': road_metadata.get('surface_type', 'paved'),
            'estimated_speed_limit': road_metadata.get('estimated_speed_limit', 30),
            'priority': road_metadata.get('priority', 3),
            'curvature': road_metadata.get('curvature', 'straight'),
            'can_turn_left': road_metadata.get('can_turn_left', True),
            'can_turn_right': road_metadata.get('can_turn_right', True),
            'can_go_straight': road_metadata.get('can_go_straight', True),
            'has_median': road_metadata.get('has_median', False),
            'parking_available': road_metadata.get('parking_available', True),
            
            # Copy position and flow info
            'position_flow': road_metadata.get('position_flow', 'unknown'),
            'simple_direction': road_metadata.get('simple_direction', 'unknown'),
            'start_area': road_metadata.get('start_area', 'unknown'),
            'end_area': road_metadata.get('end_area', 'unknown'),
            'geometric_angle': road_metadata.get('geometric_angle', 0),
            
            # Copy edge analysis
            'edge_analysis': road_metadata.get('edge_analysis', {}),
            
            # Lane-specific properties
            'lane_type': 'traffic_flow'
        }
        
        lane_tree = {
            'lane_id': f"road_{road_id}_{traffic_direction}_lane",
            'road_id': road_id,
            'direction': traffic_direction,  # Geographic direction
            'lane_type': 'traffic_flow',
            'points': trunk_points,
            'branches': [],
            'metadata': lane_metadata  # Complete metadata
        }
        
        # Add branches if connected to intersection
        if intersection_connection:
            branches = self.find_clean_branches_for_trimmed_road(road_id, traffic_direction, intersection_connection)
            lane_tree['branches'] = branches
            lane_tree['intersection_connection'] = intersection_connection
        
        return lane_tree

    def build_road_connections_for_zones(self):
        """Build road-intersection zone connections"""
        print("🔗 BUILDING ROAD-INTERSECTION ZONE CONNECTIONS...")
        
        self.road_connections = {}
        
        for intersection in self.intersections:
            int_id = intersection['id']
            int_center = intersection['center']
            zone = intersection.get('zone', {})
            zone_radius = zone.get('radius', 80)
            
            connected_roads = []
            
            print(f"  📍 Processing intersection zone {int_id} (radius: {zone_radius})")
            
            for road in self.roads:
                road_id = road['id']
                points = road['points']
                
                if not points:
                    continue
                
                # Find the closest approach point of the road to intersection
                closest_point, closest_distance, closest_index = self.find_closest_approach(
                    points, int_center)
                
                if closest_distance <= zone_radius * 1.2:  # Allow some tolerance
                    
                    # Determine connection type
                    start_dist = sqrt((points[0][0] - int_center[0])**2 + (points[0][1] - int_center[1])**2)
                    end_dist = sqrt((points[-1][0] - int_center[0])**2 + (points[-1][1] - int_center[1])**2)
                    
                    connection_type = "passes_through"
                    if start_dist <= zone_radius:
                        connection_type = "starts_from"
                    elif end_dist <= zone_radius:
                        connection_type = "ends_at"
                    
                    # Find the best connection point on zone boundary
                    connection_point = self.find_best_zone_connection_point(
                        zone, closest_point, int_center)
                    
                    road_metadata = road.get('metadata', {})
                    road_connection = {
                        'road_id': road_id,
                        'connection_type': connection_type,
                        'closest_approach_point': closest_point,
                        'closest_distance': closest_distance,
                        'zone_connection_point': connection_point,
                        'road_name': road_metadata.get('display_name', f'Road_{road_id}'),
                        'approach_index': closest_index  # Where in the road it approaches intersection
                    }
                    
                    connected_roads.append(road_connection)
                    print(f"    ✅ Road {road_id}: {connection_type}, dist={closest_distance:.1f}")
            
            self.road_connections[int_id] = connected_roads
            print(f"  📍 Intersection zone {int_id}: {len(connected_roads)} connected roads")

    def find_closest_approach(self, road_points, center):
        """Find the closest point on road to intersection center"""
        center_x, center_y = center
        min_distance = float('inf')
        closest_point = None
        closest_index = 0
        
        for i, point in enumerate(road_points):
            distance = sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                closest_index = i
        
        return closest_point, min_distance, closest_index

    def find_best_zone_connection_point(self, zone, road_approach_point, center):
        """Find best point on zone boundary for road to connect to"""
        boundary_points = zone.get('boundary_points', [])
        if not boundary_points:
            return center
        
        # Find boundary point closest to road approach point
        min_dist = float('inf')
        best_connection = center
        
        for boundary_point in boundary_points:
            dist = sqrt((boundary_point[0] - road_approach_point[0])**2 + 
                    (boundary_point[1] - road_approach_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_connection = boundary_point
        
        return best_connection


    def enrich_lanes_with_traffic_geometry(self):
        """Add traffic-aware geometric metadata to lanes for LHT navigation"""
        print("🚗 ENRICHING LANES WITH TRAFFIC GEOMETRY FOR LHT...")
        
        import math
        
        # Traffic system - Japan uses Left-Hand Traffic
        TRAFFIC_SYSTEM = "LHT"
        
        # 16-way heading labels with degrees
        HEADINGS_16_DEG = {
            "E+": 0, "NE+": 45, "N+": 90, "NW+": 135, 
            "W+": 180, "SW+": 225, "S+": 270, "SE+": 315,
            "E-": 180, "NE-": 225, "N-": 270, "NW-": 315, 
            "W-": 0, "SW-": 45, "S-": 90, "SE-": 135
        }
        
        # 8-way direction labels
        DIR8_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        
        def normalize_vector(v):
            """Normalize vector to unit length"""
            mag = math.sqrt(v[0]**2 + v[1]**2)
            return (v[0]/mag, v[1]/mag) if mag > 0 else (1, 0)
        
        def compute_heading_deg(fx, fy):
            """Compute heading in degrees (0=East, 90=North)"""
            return (math.atan2(fy, fx) * 180 / math.pi) % 360
        
        def dir8_from_heading(heading_deg):
            """Convert heading to 8-way direction"""
            idx = int(((heading_deg + 22.5) % 360) / 45)
            return DIR8_LABELS[idx]
        
        def angular_distance(angle1, angle2):
            """Compute angular distance between two angles"""
            diff = abs(angle1 - angle2)
            return min(diff, 360 - diff)
        
        # Process each lane tree
        for lane_tree in self.lane_trees:
            points = lane_tree.get('points', [])
            if len(points) < 2:
                continue
                
            lane_id = lane_tree.get('lane_id', 'unknown')
            road_id = lane_tree.get('road_id', 'unknown')
            metadata = lane_tree.get('metadata', {})
            
            # 🔥 FIX: PRESERVE existing names before adding traffic geometry
            existing_display_name = metadata.get('display_name', '')
            existing_name = metadata.get('name', '')
            existing_conversational_ids = metadata.get('conversational_identifiers', [])
            existing_user_descriptions = metadata.get('user_likely_descriptions', [])
            existing_narrative_visual = metadata.get('narrative_visual_characteristics', {})
            existing_narrative_directional = metadata.get('narrative_directional', {})
            existing_conversation_landmarks = metadata.get('conversation_landmarks', [])
            existing_route_context = metadata.get('route_context', {})
            
            # Get the traffic direction (now clean geographic direction)
            traffic_direction = lane_tree.get('direction', 'unknown')
            
            # 🔥 CRITICAL FIX: Convert points to proper coordinate system for CALCULATIONS ONLY
            # DO NOT MODIFY THE ORIGINAL POINTS - they are used for waypoints
            map_points_for_calculation = []
            for point in points:
                x, y = point[0], point[1]
                map_y = CANVAS_SIZE[1] - y  # Flip y so North is up - FOR CALCULATION ONLY
                map_points_for_calculation.append([x, map_y])
            
            # Compute forward vector using transformed coordinates (for calculations)
            start_point = map_points_for_calculation[0]
            end_point = map_points_for_calculation[-1]
            forward_vec = (end_point[0] - start_point[0], end_point[1] - start_point[1])
            forward_unit = normalize_vector(forward_vec)
            fx, fy = forward_unit
            
            # Compute driver-relative vectors
            left_unit = (-fy, fx)    # 90° counter-clockwise
            right_unit = (fy, -fx)   # 90° clockwise
            
            # Compute heading and directions
            heading_deg = compute_heading_deg(fx, fy)
            dir8 = dir8_from_heading(heading_deg)
            driver_left_dir8 = dir8_from_heading(heading_deg + 90)
            
            # Compute alignment bin (16-way)
            alignment_bin16 = int((heading_deg + 11.25) / 22.5) % 16
            
            # Compute use_when_heading mapping
            use_when_heading = {}
            for label, target_heading in HEADINGS_16_DEG.items():
                aligned = angular_distance(heading_deg, target_heading) <= 22.5
                use_when_heading[label] = aligned
            
            # Add traffic geometry to lane metadata
            traffic_geometry = {
                "traffic_system": TRAFFIC_SYSTEM,
                "traffic_direction": traffic_direction,  # Clean geographic direction
                "forward_unit": [round(fx, 4), round(fy, 4)],
                "left_unit": [round(left_unit[0], 4), round(left_unit[1], 4)],
                "right_unit": [round(right_unit[0], 4), round(right_unit[1], 4)],
                "heading_deg": round(heading_deg, 2),
                "dir8": dir8,
                "driver_left_dir8": driver_left_dir8,
                "alignment_bin16": alignment_bin16,
                "use_when_heading": use_when_heading,  # CRITICAL: Add this back
                "allowed_turns": ["through", "left", "right"],
                # 🔥 CRITICAL FIX: DO NOT STORE TRANSFORMED COORDINATES AS WAYPOINTS
                # The original lane_tree['points'] will be used for waypoints
                # "map_coordinate_points": map_points_for_calculation  # REMOVED - this was causing the issue
            }
            
            # 🔥 FIX: UPDATE metadata while preserving ALL existing names and narrative data
            metadata.update(traffic_geometry)
            
            # 🔥 FIX: Restore all preserved names and narrative metadata
            if existing_display_name:
                metadata['display_name'] = existing_display_name
            if existing_name:
                metadata['name'] = existing_name
            if existing_conversational_ids:
                metadata['conversational_identifiers'] = existing_conversational_ids
            if existing_user_descriptions:
                metadata['user_likely_descriptions'] = existing_user_descriptions
            if existing_narrative_visual:
                metadata['narrative_visual_characteristics'] = existing_narrative_visual
            if existing_narrative_directional:
                metadata['narrative_directional'] = existing_narrative_directional
            if existing_conversation_landmarks:
                metadata['conversation_landmarks'] = existing_conversation_landmarks
            if existing_route_context:
                metadata['route_context'] = existing_route_context
            
            print(f"  ✅ Lane {lane_id}: traffic_direction={traffic_direction}, heading={heading_deg:.1f}° ({dir8}), display_name='{metadata.get('display_name', 'NO_NAME')}'")
        
        # Compute lateral rankings
        self.compute_lateral_rankings()




    def compute_lateral_rankings(self):
        """Compute lateral_rank within road groups for LHT lane selection"""
        print("📊 New Logic is reflected COMPUTING LATERAL RANKINGS FOR LHT...")
        
        # Group lanes by (road_id, dir8)
        lane_groups = {}
        for lane_tree in self.lane_trees:
            metadata = lane_tree.get('metadata', {})
            road_id = lane_tree.get('road_id', 'unknown')
            dir8 = metadata.get('dir8', 'unknown')
            group_key = f"{road_id}_{dir8}"
            
            if group_key not in lane_groups:
                lane_groups[group_key] = []
            lane_groups[group_key].append(lane_tree)
        
        # Compute lateral ranking within each group
        for group_key, lanes in lane_groups.items():
            if len(lanes) <= 1:
                # Single lane gets rank 1
                lanes[0]['metadata']['lateral_rank'] = 1
                lanes[0]['metadata']['lanes_in_group'] = 1
                
                # FIXED: Update use_when_heading for single lane
                metadata = lanes[0]['metadata']
                if 'use_when_heading' in metadata:
                    for label in metadata['use_when_heading']:
                        if metadata['use_when_heading'][label]:  # If aligned
                            metadata['use_when_heading'][label] = True  # Single lane gets it
                continue
            
            # Compute center points and left projections
            lane_projections = []
            for lane_tree in lanes:
                metadata = lane_tree['metadata']
                points = metadata.get('map_coordinate_points', lane_tree.get('points', []))
                
                # Compute lane center
                if points:
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                else:
                    center_x, center_y = 0, 0
                
                # Get left unit vector
                left_unit = metadata.get('left_unit', [0, 1])
                
                # Compute leftward projection (how far left this lane is)
                leftward_score = center_x * left_unit[0] + center_y * left_unit[1]
                
                lane_projections.append((lane_tree, leftward_score, (center_x, center_y)))
            
            # Sort by leftward score (descending = leftmost first)
            lane_projections.sort(key=lambda x: x[1], reverse=True)
            
            # Assign lateral ranks
            for rank, (lane_tree, score, center) in enumerate(lane_projections, 1):
                metadata = lane_tree['metadata']
                metadata['lateral_rank'] = rank
                metadata['lanes_in_group'] = len(lanes)
                metadata['leftward_score'] = round(score, 2)
                metadata['lane_center'] = [round(center[0], 1), round(center[1], 1)]
                
                # FIXED: Update use_when_heading based on LHT rules
                if 'use_when_heading' in metadata:
                    for label in metadata['use_when_heading']:
                        if metadata['use_when_heading'][label]:  # If aligned
                            metadata['use_when_heading'][label] = (rank == 1)  # LHT: use leftmost
                
                lane_id = lane_tree.get('lane_id', 'unknown')
                dir8 = metadata.get('dir8', 'unknown')
                print(f"    🚗 Lane {lane_id}: rank {rank}/{len(lanes)} ({dir8}), leftward_score={score:.2f}")

    def add_narrative_to_traffic_mapping(self):
        """Map user narrative descriptions to traffic geometry"""
        print("💬 ADDING NARRATIVE TO TRAFFIC GEOMETRY MAPPING...")
        
        for lane_tree in self.lane_trees:
            metadata = lane_tree.get('metadata', {})
            
            # Get narrative data
            conv_ids = metadata.get('conversational_identifiers', [])
            user_descriptions = metadata.get('user_likely_descriptions', [])
            directional_narrative = metadata.get('narrative_directional', {})
            
            # Get traffic geometry
            traffic_direction = metadata.get('traffic_direction', 'unknown')
            dir8 = metadata.get('dir8', 'unknown')
            heading_deg = metadata.get('heading_deg', 0)
            lateral_rank = metadata.get('lateral_rank', 1)
            use_when_heading = metadata.get('use_when_heading', {})
            
            # Create narrative-to-geometry mapping
            narrative_traffic_mapping = {
                "lane_suitable_for_narratives": [],
                "traffic_flow_direction": traffic_direction,  # Use clean traffic_direction
                "recommended_for_heading": [],
                "user_likely_scenarios": []
            }
            
            # Map use_when_heading to narrative descriptions
            for heading_label, suitable in use_when_heading.items():
                if suitable:
                    narrative_traffic_mapping["recommended_for_heading"].append(heading_label)
                    
                    # Create user scenarios
                    direction_name = heading_label.replace('+', '').replace('-', '')
                    if '+' in heading_label:
                        scenario = f"traveling {direction_name.lower()}"
                    else:
                        scenario = f"coming from {direction_name.lower()}"
                    narrative_traffic_mapping["user_likely_scenarios"].append(scenario)
            
            # Add to metadata
            metadata['narrative_traffic_mapping'] = narrative_traffic_mapping
            
            lane_id = lane_tree.get('lane_id', 'unknown')
            suitable_count = len(narrative_traffic_mapping["recommended_for_heading"])
            print(f"  ✅ Lane {lane_id}: suitable for {suitable_count} heading scenarios")

    def ensure_metadata_consistency(self):
        """Ensure consistent metadata between roads, lanes, and what gets visualized"""
        print("🔄 ENSURING METADATA CONSISTENCY...")
        
        # Create a mapping of road_id -> full road metadata
        road_metadata_map = {}
        for road in self.roads:
            road_metadata_map[road['id']] = road.get('metadata', {})
        
        # Update lane metadata to match road metadata
        lanes_updated = 0
        for lane_tree in self.lane_trees:
            road_id = lane_tree.get('road_id')
            if road_id in road_metadata_map:
                road_meta = road_metadata_map[road_id]
                lane_meta = lane_tree.get('metadata', {})
                
                # Get current lane direction
                traffic_direction = lane_tree.get('direction', 'unknown')
                
                # 🔥 FIX: Create consistent display name
                road_display_name = road_meta.get('display_name', f'Road_{road_id}')
                
                # Check if road_display_name has ??? - if so, create clean name
                if '???' in road_display_name:
                    road_class = road_meta.get('road_class', 'local_street')
                    road_position_flow = road_meta.get('position_flow', 'unknown')
                    
                    # Generate clean name based on available metadata
                    if 'center' in road_position_flow:
                        clean_road_name = f"Central {road_class.replace('_', ' ').title()}"
                    elif 'horizontal' in road_position_flow:
                        clean_road_name = f"Horizontal {road_class.replace('_', ' ').title()}"
                    elif 'vertical' in road_position_flow:
                        clean_road_name = f"Vertical {road_class.replace('_', ' ').title()}"
                    elif 'diagonal' in road_position_flow:
                        clean_road_name = f"Diagonal {road_class.replace('_', ' ').title()}"
                    else:
                        clean_road_name = f"{road_class.replace('_', ' ').title()}"
                    
                    road_display_name = clean_road_name
                    
                    # Update the road metadata too for consistency
                    road_meta['display_name'] = clean_road_name
                    road_meta['name'] = clean_road_name
                
                consistent_display_name = f"{road_display_name} ({traffic_direction} Lane)"
                consistent_name = f"{traffic_direction} Lane of {road_display_name}"
                
                # 🔥 FIX: Update lane metadata with consistent names and preserve all data
                lane_meta.update({
                    'display_name': consistent_display_name,
                    'name': consistent_name,
                    'parent_road_display_name': road_display_name,
                    
                    # Ensure all narrative metadata matches parent road (only if not already present)
                    'conversational_identifiers': lane_meta.get('conversational_identifiers') or road_meta.get('conversational_identifiers', []),
                    'user_likely_descriptions': lane_meta.get('user_likely_descriptions') or road_meta.get('user_likely_descriptions', []),
                    'narrative_visual_characteristics': lane_meta.get('narrative_visual_characteristics') or road_meta.get('narrative_visual_characteristics', {}),
                    'narrative_directional': lane_meta.get('narrative_directional') or road_meta.get('narrative_directional', {}),
                    'conversation_landmarks': lane_meta.get('conversation_landmarks') or road_meta.get('conversation_landmarks', []),
                    'route_context': lane_meta.get('route_context') or road_meta.get('route_context', {}),
                    'nearby_landmarks': lane_meta.get('nearby_landmarks') or road_meta.get('nearby_landmarks', []),
                    'visible_labels': lane_meta.get('visible_labels') or road_meta.get('visible_labels', []),
                    'leads_to': lane_meta.get('leads_to') or road_meta.get('leads_to', {}),
                    
                    # Ensure road classification data is present
                    'road_class': lane_meta.get('road_class') or road_meta.get('road_class', 'local_street'),
                    'road_type': lane_meta.get('road_type') or road_meta.get('road_type', 'street'),
                    'width_category': lane_meta.get('width_category') or road_meta.get('width_category', 'medium'),
                    'estimated_speed_limit': lane_meta.get('estimated_speed_limit') or road_meta.get('estimated_speed_limit', 30),
                    'priority': lane_meta.get('priority') or road_meta.get('priority', 3),
                })
                
                lanes_updated += 1
                print(f"  ✅ Synchronized lane {lane_tree.get('lane_id', '?')}: '{consistent_display_name}'")
            else:
                print(f"  ⚠️  Lane {lane_tree.get('lane_id', '?')}: No matching road metadata found for road_id {road_id}")
        
        print(f"  📊 Updated {lanes_updated}/{len(self.lane_trees)} lanes for consistency")

    # ==================== UTILITY & OUTPUT METHODS ====================
    
    def make_json_serializable(self, obj):
        """Convert NumPy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.make_json_serializable(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_lane_trees(self):
        """ENHANCED: Save lane trees with verification of narrative metadata"""
        print("💾 SAVING INTEGRATED LANE TREES WITH NARRATIVE VERIFICATION...")
        
        # VERIFICATION: Check that lanes have narrative metadata
        lanes_with_conv_ids = 0
        lanes_with_user_desc = 0
        lanes_with_visual = 0
        lanes_with_directional = 0
        lanes_with_route_context = 0
        total_conv_ids = 0
        total_user_descriptions = 0
        
        print("🔍 NARRATIVE METADATA VERIFICATION:")
        for i, tree in enumerate(self.lane_trees[:3]):  # Check first 3
            metadata = tree.get('metadata', {})
            lane_id = tree.get('lane_id', f'lane_{i}')
            
            conv_ids = metadata.get('conversational_identifiers', [])
            user_desc = metadata.get('user_likely_descriptions', [])
            visual = metadata.get('narrative_visual_characteristics', {})
            directional = metadata.get('narrative_directional', {})
            route_context = metadata.get('route_context', {})
            
            print(f"  Lane {lane_id}:")
            print(f"    Conv IDs: {len(conv_ids)} - {conv_ids[:1]}")
            print(f"    User desc: {len(user_desc)} - {user_desc[:1]}")
            print(f"    Visual: {len(visual)} keys")
            print(f"    Directional: {len(directional)} keys")
            print(f"    Route context: {len(route_context)} keys")
            
            if conv_ids: lanes_with_conv_ids += 1
            if user_desc: lanes_with_user_desc += 1
            if visual: lanes_with_visual += 1
            if directional: lanes_with_directional += 1
            if route_context: lanes_with_route_context += 1
            
            total_conv_ids += len(conv_ids)
            total_user_descriptions += len(user_desc)
        
        # Calculate statistics with safe access
        total_branches = sum(len(tree.get('branches', [])) for tree in self.lane_trees)
        trees_with_branches = sum(1 for tree in self.lane_trees if tree.get('branches', []))
        
        # Count ALL narrative elements across ALL lanes
        all_conv_ids = 0
        all_user_descriptions = 0
        all_landmarks = set()
        all_businesses = set()
        all_road_names = set()
        
        for tree in self.lane_trees:
            metadata = tree.get('metadata', {})
            
            # Count narrative elements
            conv_ids = metadata.get('conversational_identifiers', [])
            user_desc = metadata.get('user_likely_descriptions', [])
            landmarks = metadata.get('conversation_landmarks', [])
            
            all_conv_ids += len(conv_ids) 
            all_user_descriptions += len(user_desc)
            
            for landmark in landmarks:
                if isinstance(landmark, dict):
                    all_landmarks.add(landmark.get('name', landmark.get('landmark_description', 'Unknown')))
                else:
                    all_landmarks.add(str(landmark))
            
            road_name = metadata.get('display_name', '')
            if road_name:
                all_road_names.add(road_name)
            
            # Add branch narrative data
            for branch in tree.get('branches', []):
                target_name = branch.get('target_road_name', '')
                if target_name:
                    all_road_names.add(target_name)
                
                target_conv_ids = branch.get('target_conversational_identifiers', [])
                target_user_desc = branch.get('target_user_likely_descriptions', [])
                all_conv_ids += len(target_conv_ids)
                all_user_descriptions += len(target_user_desc)
        
        output_data = {
            'lane_trees': self.lane_trees,
            'road_connections': getattr(self, 'road_connections', {}),
            'comprehensive_metadata': getattr(self, 'comprehensive_metadata', {}),
            'narrative_metadata': getattr(self, 'narrative_metadata', {}),  # ADD THIS
            'obstacle_analysis': getattr(self, 'obstacle_analysis', {}),
            'fragment_analysis': getattr(self, 'fragment_analysis', {}),
            
            # ENHANCED navigation metadata with narrative verification
            'navigation_metadata': {
                'all_landmarks': list(filter(None, all_landmarks)),
                'all_road_names': list(filter(None, all_road_names)),
                'intersection_types': [intersection.get('metadata', {}).get('intersection_type', 'unknown') for intersection in self.intersections],
                
                # NARRATIVE METADATA SUMMARY
                'narrative_summary': {
                    'total_conversational_identifiers': all_conv_ids,
                    'total_user_descriptions': all_user_descriptions,
                    'lanes_with_narrative': {
                        'conversational_ids': lanes_with_conv_ids,
                        'user_descriptions': lanes_with_user_desc,
                        'visual_characteristics': lanes_with_visual,
                        'directional_narrative': lanes_with_directional,
                        'route_context': lanes_with_route_context
                    }
                }
            },
            
            # Enhanced statistics with narrative verification
            'statistics': {
                'total_trees': len(self.lane_trees),
                'total_branches': total_branches,
                'trees_with_branches': trees_with_branches,
                'average_branches_per_tree': total_branches / len(self.lane_trees) if self.lane_trees else 0,
                'roads_count': len(self.roads),
                'intersections_count': len(self.intersections),
                'unique_landmarks': len(all_landmarks),
                'unique_road_names': len(all_road_names),
                
                # NARRATIVE STATISTICS
                'narrative_statistics': {
                    'total_conversational_identifiers': all_conv_ids,
                    'total_user_descriptions': all_user_descriptions,
                    'lanes_with_complete_narrative': lanes_with_conv_ids + lanes_with_user_desc + lanes_with_visual,
                    'narrative_coverage_percentage': round((lanes_with_conv_ids / len(self.lane_trees)) * 100, 2) if self.lane_trees else 0
                }
            },
            
            'parameters': {
                'lane_offset_px': LANE_OFFSET_PX,
                'intersection_radius': INTERSECTION_RADIUS
            },
            'metadata': {
                'coordinate_system': 'image_pixels',
                'origin': 'top_left',
                'supports_narrative_parsing': True,
                'supports_landmark_navigation': True,
                'supports_turn_by_turn': True,
                'uses_trimmed_roads': True,
                'uses_intersection_zones': True,
                'uses_comprehensive_metadata': True,
                'has_complete_narrative_metadata': all_conv_ids > 0 and all_user_descriptions > 0,
                'narrative_metadata_version': '2.0_complete'
            }
        }
        
        # Make JSON serializable
        output_data = self.make_json_serializable(output_data)
        
        lane_trees_filename = self.get_output_filename("lane_tree_routes_enhanced.json")
        with open(lane_trees_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.upload_output_to_s3(lane_trees_filename)
        
        print(f"  💾 Saved lane trees to {lane_trees_filename}")
        print(f"  📊 NARRATIVE VERIFICATION:")
        print(f"    🗣️  {all_conv_ids} total conversational identifiers")
        print(f"    👤 {all_user_descriptions} total user descriptions") 
        print(f"    🏷️  {lanes_with_conv_ids}/{len(self.lane_trees)} lanes have conversational IDs")
        print(f"    📋 {lanes_with_user_desc}/{len(self.lane_trees)} lanes have user descriptions")
        print(f"    🎯 Narrative coverage: {round((lanes_with_conv_ids / len(self.lane_trees)) * 100, 2)}%")

    def save_outputs(self):
        """Save outputs in both integrated and separate formats"""
        # Original integrated format with comprehensive metadata
        integrated_data = {
            "roads": self.roads,
            "intersections": self.intersections,
            "comprehensive_metadata": getattr(self, 'comprehensive_metadata', {}),
            "obstacle_analysis": {
                "ignorable_obstacles": self.ignorable_obstacles,
                "meaningful_obstacles": self.meaningful_obstacles
            },
            "fragment_analysis": {
                "ignorable_fragments": self.ignorable_fragments,
                "meaningful_fragments": self.meaningful_fragments
            },
            "metadata": {
                "total_roads": len(self.roads),
                "total_intersections": len(self.intersections),
                "obstacles_removed": len(self.ignorable_obstacles),
                "fragments_removed": len(self.ignorable_fragments),
                "has_comprehensive_metadata": True,
                "roads_trimmed_at_intersections": True
            }
        }
        
        # Make all data JSON serializable
        integrated_data = self.make_json_serializable(integrated_data)
        
        integrated_filename = self.get_output_filename("integrated_road_network.json")
        with open(integrated_filename, 'w') as f:
            json.dump(integrated_data, f, indent=2)
        self.upload_output_to_s3(integrated_filename)
        
        # Save separate centerlines
        centerlines_data = {
            "roads": self.roads,
            "comprehensive_metadata": getattr(self, 'comprehensive_metadata', {}),
            "metadata": {
                "total_roads": len(self.roads),
                "coordinate_system": "image_pixels",
                "origin": "top_left",
                "has_comprehensive_metadata": True
            }
        }
        
        centerlines_data = self.make_json_serializable(centerlines_data)
        centerlines_filename = self.get_output_filename("centerlines_with_metadata.json")
        with open(centerlines_filename, 'w') as f:
            json.dump(centerlines_data, f, indent=2)
        self.upload_output_to_s3(centerlines_filename)
        
        # Save separate intersections
        intersections_data = {
            "intersections": self.intersections,
            "metadata": {
                "total_intersections": len(self.intersections),
                "coordinate_system": "image_pixels", 
                "origin": "top_left",
                "has_comprehensive_metadata": True
            }
        }
        
        intersections_data = self.make_json_serializable(intersections_data)
        intersections_filename = self.get_output_filename("intersections_with_metadata.json")
        with open(intersections_filename, 'w') as f:
            json.dump(intersections_data, f, indent=2)
        self.upload_output_to_s3(intersections_filename)
        
        print(f"💾 Saved integrated data to {integrated_filename}")
        print(f"💾 Saved separate centerlines to {centerlines_filename}")
        print(f"💾 Saved separate intersections to {intersections_filename}")

    def draw_compass(self, canvas):
        """Draw compass rose showing N/S/E/W directions"""
        h, w = canvas.shape[:2]
        center_x, center_y = w - 80, 80
        radius = 30
        
        # Draw compass circle
        cv2.circle(canvas, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw directional arrows and labels
        directions = [
            ('N', (0, -radius+5), (255, 255, 255)),
            ('S', (0, radius-5), (255, 255, 255)),
            ('E', (radius-5, 0), (255, 255, 255)),
            ('W', (-radius+5, 0), (255, 255, 255))
        ]
        
        for label, (dx, dy), color in directions:
            cv2.putText(canvas, label, (center_x + dx - 5, center_y + dy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def visualize_integrated_network_with_comprehensive_metadata(self):
        """Enhanced visualization showing COMPLETE narrative metadata with error handling"""
        canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # FIRST: Draw intersection zones with FULL narrative metadata
        for intersection in self.intersections:
            center_x, center_y = intersection['center']
            zone = intersection.get('zone', {})
            metadata = intersection.get('metadata', {})
            
            if zone:
                radius = zone.get('radius', 80)
                
                # Color code by traffic volume
                traffic_volume = metadata.get('traffic_volume', 'medium')
                zone_color = {
                    'low': (0, 255, 0),       # Green
                    'medium': (255, 255, 0),  # Yellow  
                    'high': (255, 165, 0)     # Orange
                }.get(traffic_volume, (255, 255, 0))
                
                # Draw zone boundary
                cv2.circle(canvas, (int(center_x), int(center_y)), radius, zone_color, 3)
                
                # Semi-transparent fill
                overlay = canvas.copy()
                cv2.circle(overlay, (int(center_x), int(center_y)), radius, zone_color, -1)
                cv2.addWeighted(canvas, 0.8, overlay, 0.2, 0, canvas)
            
            # ENHANCED: Show FULL narrative metadata for intersections with safe access
            intersection_name = metadata.get('intersection_name', f'Intersection_{intersection["id"]}')
            
            # Get ALL narrative identifiers with safe access
            conversational_ids = metadata.get('conversational_identifiers', [])
            user_descriptions = metadata.get('user_likely_descriptions', [])
            navigation_conversations = metadata.get('navigation_conversations', {})
            conversation_landmarks = metadata.get('conversation_landmarks', [])
            route_decision_context = metadata.get('route_decision_context', {})
            
            cv2.circle(canvas, (int(center_x), int(center_y)), 8, (255, 255, 255), -1)
            
            # COMPREHENSIVE LABELING: Show ALL narrative data with safe access
            label_y = int(center_y) - 60
            line_height = 12
            
            # Main intersection name
            cv2.putText(canvas, intersection_name, 
                    (int(center_x) + 30, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show conversational identifiers safely
            if conversational_ids and len(conversational_ids) > 0:
                conv_text = f"Users say: '{conversational_ids[0]}'"
                if len(conversational_ids) > 1:
                    conv_text += f", '{conversational_ids[1]}'"
                cv2.putText(canvas, conv_text,
                        (int(center_x) + 30, label_y + line_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
            
            # Show user likely descriptions safely
            if user_descriptions and len(user_descriptions) > 0:
                desc_text = f"Described as: '{user_descriptions[0]}'"
                cv2.putText(canvas, desc_text,
                        (int(center_x) + 30, label_y + line_height * 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
            
            # Show navigation conversations safely
            nav_convs = navigation_conversations.get('from_here_you_can', [])
            if nav_convs and len(nav_convs) > 0:
                nav_text = f"Navigation: '{nav_convs[0][:30]}...'"
                cv2.putText(canvas, nav_text,
                        (int(center_x) + 30, label_y + line_height * 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)
            
            # Show conversation landmarks safely
            if conversation_landmarks and len(conversation_landmarks) > 0:
                landmark = conversation_landmarks[0]
                if isinstance(landmark, dict):
                    landmark_text = f"Near: {landmark.get('name', landmark.get('landmark_description', 'Unknown'))}"
                else:
                    landmark_text = f"Near: {str(landmark)}"
                cv2.putText(canvas, landmark_text,
                        (int(center_x) + 30, label_y + line_height * 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1)
            
            # Show route decision context safely
            decision_context = route_decision_context.get('why_users_mention_it', '')
            if decision_context:
                context_text = f"Context: {decision_context[:25]}..."
                cv2.putText(canvas, context_text,
                        (int(center_x) + 30, label_y + line_height * 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 255), 1)
        
        # SECOND: Draw roads with COMPREHENSIVE narrative metadata
        for i, road in enumerate(self.roads):
            color = colors[i % len(colors)]
            points = road["points"]
            metadata = road.get('metadata', {})
            
            # Vary thickness based on road class
            road_class = metadata.get('road_class', 'local_street')
            thickness_map = {
                'major_highway': 6,
                'main_street': 5,
                'secondary_road': 4,
                'local_street': 3
            }
            thickness = thickness_map.get(road_class, 3)
            
            if len(points) > 1:
                pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
            
            # COMPREHENSIVE ROAD LABELING: Show ALL narrative metadata safely
            if points:
                mid_idx = len(points) // 2
                mid_pt = (int(points[mid_idx][0]), int(points[mid_idx][1]))
                
                # Get ALL narrative data safely
                display_name = metadata.get('display_name', f'Road_{i}')
                conversational_ids = metadata.get('conversational_identifiers', [])
                user_descriptions = metadata.get('user_likely_descriptions', [])
                narrative_visual = metadata.get('narrative_visual_characteristics', {})
                narrative_directional = metadata.get('narrative_directional', {})
                conversation_landmarks = metadata.get('conversation_landmarks', [])
                route_context = metadata.get('route_context', {})
                leads_to = metadata.get('leads_to', {})
                
                # Calculate label position to avoid overlap
                label_x = mid_pt[0] - 100
                label_y = mid_pt[1] - 50
                line_height = 12
                
                # Main road name with conversational identifier
                main_label = display_name
                if conversational_ids and len(conversational_ids) > 0:
                    main_label += f" ('{conversational_ids[0]}')"
                
                cv2.putText(canvas, main_label, 
                        (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                
                # User likely descriptions safely
                if user_descriptions and len(user_descriptions) > 0:
                    user_desc = f"Users say: '{user_descriptions[0]}'"
                    cv2.putText(canvas, user_desc, 
                            (label_x, label_y + line_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
                
                # Visual characteristics safely
                width_desc = narrative_visual.get('width_description', '')
                surface_desc = narrative_visual.get('surface_appearance', '')
                if width_desc or surface_desc:
                    visual_text = f"Looks: {width_desc} {surface_desc}".strip()
                    cv2.putText(canvas, visual_text, 
                            (label_x, label_y + line_height * 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
                
                # Directional narrative safely
                where_from = narrative_directional.get('where_it_comes_from', '')
                where_to = narrative_directional.get('where_it_goes_to', '')
                if where_from or where_to:
                    direction_text = f"From: {where_from} To: {where_to}"
                    if len(direction_text) > 35:
                        direction_text = direction_text[:35] + "..."
                    cv2.putText(canvas, direction_text, 
                            (label_x, label_y + line_height * 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 200), 1)
                
                # Route context - entry/exit points safely
                entry_points = route_context.get('entry_points', [])
                exit_points = route_context.get('exit_points', [])
                common_destinations = route_context.get('common_destinations', [])
                
                if entry_points and len(entry_points) > 0:
                    entry_text = f"Entry: {entry_points[0]}"
                    cv2.putText(canvas, entry_text, 
                            (label_x, label_y + line_height * 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1)
                
                if exit_points and len(exit_points) > 0:
                    exit_text = f"Exit: {exit_points[0]}"
                    cv2.putText(canvas, exit_text, 
                            (label_x, label_y + line_height * 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 255), 1)
                
                if common_destinations and len(common_destinations) > 0:
                    dest_text = f"Goes to: {common_destinations[0]}"
                    cv2.putText(canvas, dest_text, 
                            (label_x, label_y + line_height * 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 255), 1)
                
                # Conversation landmarks safely
                if conversation_landmarks and len(conversation_landmarks) > 0:
                    landmark = conversation_landmarks[0]
                    if isinstance(landmark, dict):
                        landmark_name = landmark.get('name', landmark.get('landmark_description', str(landmark)))
                    else:
                        landmark_name = str(landmark)
                    landmark_text = f"Near: {landmark_name}"
                    cv2.putText(canvas, landmark_text, 
                            (label_x, label_y + line_height * 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Enhanced title with COMPREHENSIVE narrative statistics
        narrative_metadata = getattr(self, 'narrative_metadata', {})
        road_narratives = narrative_metadata.get('road_narratives', [])
        intersection_narratives = narrative_metadata.get('intersection_narratives', [])
        area_context = narrative_metadata.get('area_context', {})
        route_patterns = narrative_metadata.get('common_route_patterns', [])
        
        # Count narrative elements safely
        total_conversational_ids = 0
        total_user_descriptions = 0
        total_landmarks = 0
        
        for road in road_narratives:
            total_conversational_ids += len(road.get('conversational_identifiers', []))
            total_user_descriptions += len(road.get('user_likely_descriptions', []))
            total_landmarks += len(road.get('landmark_references', []))
        
        title = f"COMPREHENSIVE NARRATIVE NETWORK: {len(self.roads)} roads, {len(self.intersections)} intersections"
        narrative_stats = f"Narrative: {total_conversational_ids} conversation IDs, {total_user_descriptions} user descriptions, {total_landmarks} landmark refs"
        context_stats = f"Context: {len(area_context)} areas, {len(route_patterns)} route patterns | Full metadata displayed"
        
        cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, narrative_stats, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        cv2.putText(canvas, context_stats, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
        
        # Add legend for narrative elements
        legend_y = 100
        legend_items = [
            ("White: Names & Conversational IDs", (255, 255, 255)),
            ("Green: User Descriptions", (200, 255, 200)),
            ("Blue: Visual Characteristics", (200, 200, 255)),
            ("Red: Directional Narrative", (255, 200, 200)),
            ("Yellow: Entry/Exit Points", (255, 255, 200)),
            ("Magenta: Destinations", (255, 200, 255)),
            ("Cyan: Context & Landmarks", (200, 255, 255))
        ]
        
        for i, (legend_text, color) in enumerate(legend_items):
            cv2.putText(canvas, legend_text, (20, legend_y + i * 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Add compass
        self.draw_compass(canvas)
        
        # Save with narrative suffix
        comprehensive_network_viz_path = "comprehensive_narrative_network_visualization.png"
        cv2.imwrite(comprehensive_network_viz_path, canvas)
        self.upload_output_to_s3(comprehensive_network_viz_path)
        
        integrated_viz_filename = self.get_output_filename("integrated_network_visualization.png")
        cv2.imwrite(integrated_viz_filename, canvas)
        self.upload_output_to_s3(integrated_viz_filename)
        
        print(f"  💾 Saved COMPREHENSIVE narrative network visualization to {comprehensive_network_viz_path}")
        print(f"  📊 Visualization includes FULL narrative metadata: {total_conversational_ids} IDs, {total_user_descriptions} descriptions, {total_landmarks} landmarks")

    def visualize_integrated_lanes_with_comprehensive_metadata(self):
        """COMPREHENSIVE lane visualization with ALL narrative metadata and error handling"""
        print("🎨 CREATING COMPREHENSIVE LANE VISUALIZATION WITH FULL NARRATIVE METADATA...")
        
        if not self.lane_trees:
            print("  ⚠️  No lane trees available for visualization")
            return
        
        # Create larger canvas for more text
        canvas = np.zeros((CANVAS_SIZE[1] + 200, CANVAS_SIZE[0] + 300, 3), dtype=np.uint8)
        
        # Colors for different lane types and road classes
        ROAD_CLASS_COLORS = {
            'major_highway': (255, 100, 100),   # Light red
            'main_street': (100, 255, 100),     # Light green
            'secondary_road': (100, 100, 255),  # Light blue
            'local_street': (255, 255, 100)     # Light yellow
        }
        
        # FIRST: Draw intersection zones (same as before but with error handling)
        for intersection in self.intersections:
            center_x, center_y = intersection['center']
            zone = intersection.get('zone', {})
            metadata = intersection.get('metadata', {})
            
            if zone:
                radius = zone.get('radius', 80)
                traffic_volume = metadata.get('traffic_volume', 'medium')
                zone_color = {
                    'low': (0, 255, 0),
                    'medium': (255, 255, 0),  
                    'high': (255, 165, 0)
                }.get(traffic_volume, (255, 255, 0))
                
                cv2.circle(canvas, (int(center_x), int(center_y)), radius, zone_color, 3)
                
                overlay = canvas.copy()
                cv2.circle(overlay, (int(center_x), int(center_y)), radius, zone_color, -1)
                cv2.addWeighted(canvas, 0.8, overlay, 0.2, 0, canvas)
        
        # SECOND: Draw lanes with COMPREHENSIVE narrative metadata
        for lane_tree in self.lane_trees:
            points = lane_tree.get('points', [])
            direction = lane_tree.get('direction', 'forward')
            metadata = lane_tree.get('metadata', {})
            road_class = metadata.get('road_class', 'local_street')
            
            if len(points) < 2:
                continue
            
            # Choose color based on road class and direction
            base_color = ROAD_CLASS_COLORS.get(road_class, (255, 255, 255))
            
            if direction == 'forward':
                color = base_color
                thickness = 5
            else:
                color = tuple(int(c * 0.7) for c in base_color)
                thickness = 4
            
            # Special color for edge-connected lanes
            edge_analysis = metadata.get('edge_analysis', {})
            if edge_analysis.get('has_edge_connection', False):
                color = (0, 255, 255)  # Cyan
                thickness = 6
            
            # Draw lane
            pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
            
            # Add direction arrow
            if len(points) >= 10:
                mid_idx = len(points) // 2
                mid_pt = points[mid_idx]
                next_pt = points[min(mid_idx + 5, len(points) - 1)]
                
                cv2.arrowedLine(canvas, 
                            (int(mid_pt[0]), int(mid_pt[1])),
                            (int(next_pt[0]), int(next_pt[1])),
                            color, 3, tipLength=0.3)
            
            # COMPREHENSIVE LANE LABELING with ALL narrative metadata and safe access
            if points:
                mid_idx = len(points) // 2
                mid_pt = points[mid_idx]
                
                # Get ALL narrative metadata safely
                display_name = metadata.get('display_name', f'Road_{lane_tree["road_id"]}')
                conversational_ids = metadata.get('conversational_identifiers', [])
                user_descriptions = metadata.get('user_likely_descriptions', [])
                narrative_visual = metadata.get('narrative_visual_characteristics', {})
                narrative_directional = metadata.get('narrative_directional', {})
                conversation_landmarks = metadata.get('conversation_landmarks', [])
                route_context = metadata.get('route_context', {})
                leads_to = metadata.get('leads_to', {})
                
                # Position labels to avoid overlap
                label_x = int(mid_pt[0]) + 20
                label_y = int(mid_pt[1]) - 70 if direction == 'forward' else int(mid_pt[1]) + 30
                line_height = 11
                
                # NEW FIXED CODE:
                # No arrow needed - direction already in display_name
                main_label = display_name  # Clean, no encoding issues
                if conversational_ids and len(conversational_ids) > 0:
                    main_label += f" ('{conversational_ids[0]}')"
                
                cv2.putText(canvas, main_label, 
                        (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                
                # User likely descriptions safely
                if user_descriptions and len(user_descriptions) > 0:
                    user_desc_text = user_descriptions[0]
                    user_desc = f"Users: '{user_desc_text[:20]}...'" if len(user_desc_text) > 20 else f"Users: '{user_desc_text}'"
                    cv2.putText(canvas, user_desc, 
                            (label_x, label_y + line_height), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 255, 200), 1)
                
                # Visual characteristics safely
                width_desc = narrative_visual.get('width_description', '')
                surface_desc = narrative_visual.get('surface_appearance', '')
                if width_desc:
                    visual_text = f"Visual: {width_desc}"
                    cv2.putText(canvas, visual_text, 
                            (label_x, label_y + line_height * 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 255), 1)
                
                # Directional narrative safely
                where_from = narrative_directional.get('where_it_comes_from', '')
                where_to = narrative_directional.get('where_it_goes_to', '')
                if where_from:
                    direction_text = f"From: {where_from[:15]}"
                    cv2.putText(canvas, direction_text, 
                            (label_x, label_y + line_height * 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 200), 1)
                if where_to:
                    direction_text = f"To: {where_to[:15]}"
                    cv2.putText(canvas, direction_text, 
                            (label_x, label_y + line_height * 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 200), 1)
                
                # Route context safely
                entry_points = route_context.get('entry_points', [])
                exit_points = route_context.get('exit_points', [])
                common_destinations = route_context.get('common_destinations', [])
                
                if entry_points and len(entry_points) > 0:
                    entry_text = f"Entry: {entry_points[0][:15]}"
                    cv2.putText(canvas, entry_text, 
                            (label_x, label_y + line_height * 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 200), 1)
                
                if common_destinations and len(common_destinations) > 0:
                    dest_text = f"Dest: {common_destinations[0][:15]}"
                    cv2.putText(canvas, dest_text, 
                            (label_x, label_y + line_height * 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 255), 1)
                
                # Show branches with narrative context
                branches = lane_tree.get('branches', [])
                if branches:
                    branch_count_text = f"Branches: {len(branches)}"
                    cv2.putText(canvas, branch_count_text, 
                            (label_x, label_y + line_height * 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 255, 255), 1)
                    
                    # Show first branch with narrative
                    first_branch = branches[0]
                    target_name = first_branch.get('target_road_name', 'Unknown')
                    turn_type = first_branch.get('turn_type', 'turn')
                    branch_text = f"→ {turn_type} to {target_name[:12]}"
                    cv2.putText(canvas, branch_text, 
                            (label_x, label_y + line_height * 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 255, 255), 1)

                # 🚗 ADD TRAFFIC GEOMETRY INFO HERE (after branch info)
                # Show traffic geometry info safely
                traffic_system = metadata.get('traffic_system', 'unknown')
                dir8 = metadata.get('dir8', 'unknown')
                lateral_rank = metadata.get('lateral_rank', 0)
                lanes_in_group = metadata.get('lanes_in_group', 1)
                
                # Get narrative scenarios count
                narrative_traffic = metadata.get('narrative_traffic_mapping', {})
                heading_scenarios = len(narrative_traffic.get('recommended_for_heading', []))
                
                if traffic_system != 'unknown':
                    traffic_info = f"{traffic_system} Rank {lateral_rank}/{lanes_in_group} ({dir8}) - {heading_scenarios} scenarios"
                    cv2.putText(canvas, traffic_info, 
                            (label_x, label_y + line_height * 9), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 255, 255), 1)
                    
                    # Show recommended heading scenarios if available
                    recommended_headings = narrative_traffic.get('recommended_for_heading', [])
                    if recommended_headings:
                        heading_text = f"Good for: {', '.join(recommended_headings[:3])}"  # Show first 3
                        cv2.putText(canvas, heading_text, 
                                (label_x, label_y + line_height * 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 200, 255), 1)
                    
                    # Show user scenarios if available  
                    user_scenarios = narrative_traffic.get('user_likely_scenarios', [])
                    if user_scenarios:
                        scenario_text = f"Scenarios: {user_scenarios[0][:20]}..."
                        cv2.putText(canvas, scenario_text, 
                                (label_x, label_y + line_height * 11), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 150, 255), 1)
        
        # COMPREHENSIVE statistics panel on the right side
        stats_x = CANVAS_SIZE[0] + 20
        stats_y = 50
        line_height = 20
        
        # Background for stats panel
        cv2.rectangle(canvas, (stats_x - 10, 30), (canvas.shape[1] - 10, canvas.shape[0] - 30), (30, 30, 30), -1)
        cv2.rectangle(canvas, (stats_x - 10, 30), (canvas.shape[1] - 10, canvas.shape[0] - 30), (100, 100, 100), 2)
        
        # Title
        cv2.putText(canvas, "COMPREHENSIVE NARRATIVE + TRAFFIC", 
                (stats_x, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        stats_y += 30
        
        # Count all narrative elements safely
        narrative_metadata = getattr(self, 'narrative_metadata', {})
        road_narratives = narrative_metadata.get('road_narratives', [])
        intersection_narratives = narrative_metadata.get('intersection_narratives', [])
        
        total_conversational_ids = 0
        total_user_descriptions = 0
        total_landmarks = 0
        total_branches = 0
        
        # 🚗 ADD TRAFFIC GEOMETRY STATISTICS
        lanes_with_traffic_geometry = 0
        total_heading_scenarios = 0
        
        for lane in self.lane_trees:
            metadata = lane.get('metadata', {})
            total_conversational_ids += len(metadata.get('conversational_identifiers', []))
            total_user_descriptions += len(metadata.get('user_likely_descriptions', []))
            total_landmarks += len(metadata.get('conversation_landmarks', []))
            total_branches += len(lane.get('branches', []))
            
            # Count traffic geometry
            if metadata.get('traffic_system'):
                lanes_with_traffic_geometry += 1
            narrative_traffic = metadata.get('narrative_traffic_mapping', {})
            total_heading_scenarios += len(narrative_traffic.get('recommended_for_heading', []))
        
        # Display comprehensive stats with traffic info
        stats = [
            f"Total Lanes: {len(self.lane_trees)}",
            f"Total Branches: {total_branches}",
            f"Conversational IDs: {total_conversational_ids}",
            f"User Descriptions: {total_user_descriptions}",
            f"Landmark References: {total_landmarks}",
            "",
            "TRAFFIC GEOMETRY:",  # 🚗 ADD TRAFFIC SECTION
            f"Lanes with LHT Geometry: {lanes_with_traffic_geometry}",
            f"Total Heading Scenarios: {total_heading_scenarios}",
            f"Avg Scenarios/Lane: {total_heading_scenarios/len(self.lane_trees):.1f}" if self.lane_trees else "0",
            "",
            "NARRATIVE ELEMENTS:",
            f"Road Narratives: {len(road_narratives)}",
            f"Intersection Narratives: {len(intersection_narratives)}",
            f"Area Contexts: {len(narrative_metadata.get('area_context', {}))}",
            f"Route Patterns: {len(narrative_metadata.get('common_route_patterns', []))}",
            "",
            "METADATA COVERAGE:",
        ]
        
        # Add metadata coverage stats safely
        lanes_with_conv_ids = 0
        lanes_with_user_desc = 0
        lanes_with_visual = 0
        lanes_with_directional = 0
        lanes_with_route_context = 0
        
        for lane in self.lane_trees:
            metadata = lane.get('metadata', {})
            if metadata.get('conversational_identifiers'):
                lanes_with_conv_ids += 1
            if metadata.get('user_likely_descriptions'):
                lanes_with_user_desc += 1
            if metadata.get('narrative_visual_characteristics'):
                lanes_with_visual += 1
            if metadata.get('narrative_directional'):
                lanes_with_directional += 1
            if metadata.get('route_context'):
                lanes_with_route_context += 1
        
        coverage_stats = [
            f"Conv IDs: {lanes_with_conv_ids}/{len(self.lane_trees)}",
            f"User Desc: {lanes_with_user_desc}/{len(self.lane_trees)}",
            f"Visual: {lanes_with_visual}/{len(self.lane_trees)}",
            f"Directional: {lanes_with_directional}/{len(self.lane_trees)}",
            f"Route Context: {lanes_with_route_context}/{len(self.lane_trees)}",
            f"Traffic Geometry: {lanes_with_traffic_geometry}/{len(self.lane_trees)}",  # 🚗 ADD THIS
        ]
        
        stats.extend(coverage_stats)
        
        for stat in stats:
            if stat == "":
                stats_y += line_height // 2
                continue
            
            color = (255, 255, 255) if stat.endswith(":") else (200, 200, 200)
            font_scale = 0.4 if stat.endswith(":") else 0.35
            
            cv2.putText(canvas, stat, (stats_x, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            stats_y += line_height
        
        # Color legend
        stats_y += 20
        cv2.putText(canvas, "COLOR LEGEND:", (stats_x, stats_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        stats_y += 20
        
        legend_items = [
            ("White: Names & IDs", (255, 255, 255)),
            ("Green: User Descriptions", (200, 255, 200)),
            ("Blue: Visual Characteristics", (200, 200, 255)),
            ("Red: Directional Narrative", (255, 200, 200)),
            ("Yellow: Entry Points", (255, 255, 200)),
            ("Magenta: Destinations", (255, 200, 255)),
            ("Cyan: Branches & Context", (200, 255, 255)),
            ("Light Cyan: Traffic Geometry", (100, 255, 255)),  # 🚗 ADD THIS
        ]
        
        for legend_text, color in legend_items:
            cv2.putText(canvas, legend_text, (stats_x, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
            stats_y += 15
        
        # Main title
        title = f"COMPREHENSIVE LANE NETWORK: {len(self.lane_trees)} lanes with NARRATIVE + TRAFFIC metadata"
        cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save comprehensive lane visualization
        comprehensive_lane_viz_path = "comprehensive_narrative_lane_visualization.png"
        cv2.imwrite(comprehensive_lane_viz_path, canvas)
        self.upload_output_to_s3(comprehensive_lane_viz_path)
        
        print(f"  💾 Saved COMPREHENSIVE narrative lane visualization to {comprehensive_lane_viz_path}")
        print(f"  📊 Visualization shows ALL elements: {total_conversational_ids} IDs, {total_user_descriptions} descriptions, {lanes_with_traffic_geometry} traffic-enabled lanes")

    # ==================== MAIN INTEGRATION METHOD ====================
    def process_integrated_network_with_comprehensive_metadata_and_clean_trimming(self):
        """INTEGRATED: Best of both approaches - Clean trimming + Comprehensive metadata"""
        print("🚗 INTEGRATED ROAD NETWORK: CLEAN TRIMMING + COMPREHENSIVE METADATA")
        print("=" * 90)

        # STEP 0: Clean obstacle removal (from clean trimming approach)
        print("\n🎯 STEP 0: LLM obstacle removal...")
        cleaned_mask = self.smart_obstacle_removal_pipeline(MASK_PATH)

        # STEP 0.5: LLM network structure analysis (from clean trimming approach)
        print("\n🔢 STEP 0.5: LLM network structure analysis...")
        
        # 🔥 FIX: Save the cleaned mask numpy array to a temporary file for LLM analysis
        temp_cleaned_mask_path = self.get_output_filename("temp_cleaned_mask_for_llm_analysis.png")
        cv2.imwrite(temp_cleaned_mask_path, cleaned_mask)
        self.upload_output_to_s3(temp_cleaned_mask_path)
        
        network_analysis = self.analyze_expected_road_and_intersection_count_with_bedrock(
            temp_cleaned_mask_path, ROADMAP_PATH, SATELLITE_PATH)
        expected_roads = network_analysis.get('expected_main_roads', 4)
        expected_intersections = network_analysis.get('expected_major_intersections', 1)

        # STEP 1: Clean skeleton processing (from clean trimming approach)
        print("\n📍 STEP 1: Extracting and cleaning skeleton...")
        # 🔥 FIX: Use the cleaned_mask numpy array directly instead of file
        initial_skeleton = self.extract_medial_axis(cleaned_mask)  # Pass numpy array
        cleaned_skeleton = self.smart_fragment_cleanup_pipeline(initial_skeleton, expected_roads)   
        
        # STEP 2: Clean road tracing (from clean trimming approach)
        print("\n🛣️  STEP 2: Tracing roads from cleaned skeleton...")
        self.roads, endpoints, junctions = self.trace_skeleton_paths(cleaned_skeleton)
        
        # Handle any excess roads
        if len(self.roads) > expected_roads + 2:
            print(f"\n🔧 STEP 2.5: Road consolidation (optional)...")
            # Could add consolidation here if needed
        
        # STEP 3: Clean intersection zone creation (from clean trimming approach)
        print(f"\n🚦 STEP 3: Creating intersection zones...")
        initial_intersections = self.find_major_intersections_with_zones(junctions, cleaned_skeleton)
        self.intersections = self.consolidate_intersections(
            initial_intersections, expected_intersections, INTERSECTION_RADIUS)
        
        # STEP 3.5: CRITICAL - Road trimming at intersections (from clean trimming approach)
        print(f"\n✂️  STEP 3.5: Trimming roads at intersection boundaries...")
        self.trim_roads_at_intersection_zones()

        # STEP 4: NARRATIVE METADATA GENERATION 
        print("\n💬 STEP 4: Generating narrative metadata for route matching...")
        self.narrative_metadata = self.generate_narrative_metadata_for_route_matching(ROADMAP_PATH, SATELLITE_PATH)
        
        # STEP 5: NARRATIVE METADATA ASSIGNMENT TO ROADS
        print("\n🎯 STEP 5: Assigning narrative metadata to trimmed roads...")  
        self.assign_narrative_metadata_to_roads()
        
        print("\n🚦 STEP 6: Assigning narrative metadata to intersections...")
        self.assign_narrative_metadata_to_intersections()
        
        # STEP 7: CLEAN road connections 
        print("\n🔗 STEP 7: Building road-intersection zone connections...")
        self.build_road_connections_for_zones()
        
        # STEP 8: Save comprehensive outputs (roads + intersections with metadata)
        print("\n💾 STEP 8: Saving comprehensive outputs...")
        self.save_outputs()
        
        # STEP 9: INTEGRATED visualization with comprehensive metadata
        print("\n🎨 STEP 9: Creating integrated network visualization...")
        self.visualize_integrated_network_with_comprehensive_metadata()
                    
        print("\n🛤️  STEP 10: Generating traffic-flow based lanes...")
        self.generate_geographic_lanes_from_roads()

        print("\n🚗 STEP 10.5: Enriching lanes with traffic geometry for LHT...")
        self.enrich_lanes_with_traffic_geometry()

        print("\n💬 STEP 10.7: Adding narrative to traffic mapping...")
        self.add_narrative_to_traffic_mapping()

        print("\n🚗 STEP 10.8: Adding LHT lane usage guidance...")
        self.add_lht_lane_usage_guidance()

        print("\n🔄 STEP 10.9: Ensuring metadata consistency...")
        self.ensure_metadata_consistency()

        print("\n💾 STEP 11: Saving comprehensive lane trees...")
        self.save_lane_trees()

        print("\n🎨 STEP 12: Creating integrated lane visualization...")
        self.visualize_integrated_lanes_with_comprehensive_metadata()
        
        # NEW STEP 13: Save metadata-only JSON files for LLM
        print("\n📄 STEP 13: Saving metadata-only JSON files for LLM consumption...")
        self.save_metadata_only_outputs()
                
        print(f"\n🎉 INTEGRATED PROCESSING COMPLETE!")
        print(f"✅ Clean Network Structure:")
        print(f"   🛣️  {len(self.roads)} trimmed roads with comprehensive metadata")
        print(f"   🚦 {len(self.intersections)} intersection zones with contextual information")
        print(f"   🛤️  {len(self.lane_trees)} clean lane trees with enhanced branches")
        
        print(f"✅ Comprehensive Metadata:")
        metadata_summary = getattr(self, 'comprehensive_metadata', {})
        print(f"   📋 {len(metadata_summary.get('road_labels', []))} road labels identified")
        print(f"   🏢 {len(metadata_summary.get('landmarks', []))} landmarks mapped") 
        print(f"   🗺️  {len(metadata_summary.get('road_classifications', []))} road classifications")
        
        print(f"✅ Integration Quality:")
        total_branches = sum(len(tree.get('branches', [])) for tree in self.lane_trees)
        lanes_with_metadata = sum(1 for tree in self.lane_trees if tree.get('metadata', {}))
        print(f"   🌟 {total_branches} navigation branches generated")
        print(f"   🏷️  {lanes_with_metadata}/{len(self.lane_trees)} lanes have comprehensive metadata")
        
        obstacles_removed = len(getattr(self, 'ignorable_obstacles', []))
        fragments_removed = len(getattr(self, 'ignorable_fragments', []))
        print(f"   🧹 {obstacles_removed} obstacles + {fragments_removed} fragments removed")
        
        print(f"\n🏆 RESULT: Best of both approaches successfully integrated!")
        print(f"   • Clean, well-organized road network structure")
        print(f"   • Rich contextual metadata for navigation")  
        print(f"   • Proper intersection zones and road trimming")
        print(f"   • Enhanced lane trees with comprehensive branches")



def main():
    if len(sys.argv) < 2:
        print("❌ connection_id argument is required")
        sys.exit(1)

    connection_id = sys.argv[1]
    bucket_name = os.environ.get("BUCKET_NAME")
    
    if not bucket_name:
        print("❌ BUCKET_NAME environment variable is required")
        sys.exit(1)
    
    print(f"🚗 Starting road network generation for connection_id: {connection_id}")
    print(f"🪣 Using S3 bucket: {bucket_name}")
    
    generator = IntegratedRoadNetworkGenerator(connection_id, bucket_name)
    generator.process_integrated_network_with_comprehensive_metadata_and_clean_trimming()

if __name__ == '__main__':
    main()