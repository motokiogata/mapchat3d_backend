Looking at the PDF and your code, I need to add user-friendly aliases and narrative relationships to create a "common language between AI and human" as described in Step 0 of the document. Here's the modified code with comprehensive user-friendly metadata:

```python
import cv2
import numpy as np
import json
import boto3
import sys
import base64
import os
from io import BytesIO
from PIL import Image
from skimage.morphology import medial_axis
from collections import defaultdict
import math
import re
from math import atan2, degrees, sqrt, cos, sin, radians
from shapely.geometry import LineString, Point

# AWS Bedrock setup
BEDROCK_MODEL_ID = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_REGION = "ap-northeast-1"

# Paths
MASK_PATH = "final_road_mask_cleaned.png"
ROADMAP_PATH = "roadmap.png"
SATELLITE_PATH = "satellite.png"

# Output files - INTEGRATED
OUTPUT_INTEGRATED_JSON = "integrated_road_network.json"
OUTPUT_CENTERLINES_JSON = "centerlines_with_metadata.json"
OUTPUT_INTERSECTIONS_JSON = "intersections_with_metadata.json"
OUTPUT_LANE_TREES_JSON = "lane_tree_routes_enhanced.json"
OUTPUT_IMG = "integrated_network_visualization.png"
DEBUG_SKELETON = "debug_skeleton.png"

# Parameters
MIN_LINE_LENGTH = 20
CANVAS_SIZE = (1280, 1280)
EDGE_TOLERANCE = 10
LANE_OFFSET_PX = 20
INTERSECTION_RADIUS = 30

class IntegratedRoadNetworkGenerator:
    def __init__(self, connection_id=None):
        # Core data with consistent IDs
        self.roads = []
        self.intersections = []
        self.lane_trees = []
        
        # ID mapping and consistency
        self.road_id_counter = 0
        self.intersection_id_counter = 0
        self.lane_id_counter = 0
        
        # Cross-reference mappings
        self.road_to_intersections = {}  # road_id -> [intersection_ids]
        self.intersection_to_roads = {}  # intersection_id -> [road_ids]
        self.lane_to_road = {}  # lane_id -> road_id
        
        # Enhanced metadata
        self.bedrock_metadata = {}
        self.edge_analysis_summary = {}
        self.navigation_graph = {}
        
        # Edge analysis
        self.edge_entry_points = {}
        self.geographic_road_map = {'west': [], 'east': [], 'north': [], 'south': []}
        
        # NEW: Common language dictionaries for user-friendly aliases
        self.common_language_vocabulary = {
            'roads': {},      # road_id -> user_friendly_aliases
            'intersections': {}, # intersection_id -> user_friendly_aliases
            'lanes': {},      # lane_id -> user_friendly_aliases
            'landmarks': {},   # landmark_name -> detailed_info
            'spatial_relationships': {}  # relationships between entities
        }
        
        # S3 setup
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ.get("BUCKET_NAME", "your-output-bucket")
        if connection_id:
            self.connection_id = connection_id
        else:
            self.connection_id = os.environ.get("CONNECTION_ID", "default_connection")

    def upload_to_s3(self, local_file_path, s3_key):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            print(f"✅ Uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            print(f"❌ Failed to upload {local_file_path} to S3: {e}")
    
    def upload_json_to_s3(self, data, s3_key):
        """Upload JSON data directly to S3"""
        try:
            json_string = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_string.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Uploaded JSON data to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            print(f"❌ Failed to upload JSON to S3: {e}")

    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for Bedrock API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_roadmap_with_bedrock(self, roadmap_path, satellite_path):
        """Analyze images with AWS Bedrock Claude for user-friendly aliases"""
        bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
        
        roadmap_b64 = self.encode_image_to_base64(roadmap_path)
        satellite_b64 = self.encode_image_to_base64(satellite_path)
        
        prompt = """
        Please analyze these roadmap and satellite images to create user-friendly descriptions and spatial relationships. I need to build a "common language" between AI and humans for navigation.

        Generate comprehensive information about:

        1. USER-FRIENDLY ROAD DESCRIPTIONS
        - Natural names and descriptions (like "the main street", "narrow road behind the station")
        - Characteristics people would notice ("wide road", "curved street", "busy avenue")
        - Relative positions ("road on the left side", "street running north-south")

        2. USER-FRIENDLY INTERSECTION DESCRIPTIONS  
        - Natural references ("intersection near the station", "crossing by the supermarket")
        - Visual landmarks ("the big intersection", "T-junction with traffic lights")
        - Business references ("crossroads by Ozeki", "junction near the convenience store")

        3. LANE-LEVEL DESCRIPTIONS
        - Direction-specific lanes ("westbound lane of main road", "left turn lane at intersection")
        - Landmark references ("lane in front of the shop", "right lane heading towards station")

        4. SPATIAL RELATIONSHIPS & NARRATIVE CONNECTIONS
        - How roads connect ("main road connects to station road at the central intersection")
        - Sequential descriptions ("from the narrow street, turn right at the big intersection, then straight on the main road")
        - Landmark-based navigation ("the road that goes past the supermarket connects to the station area")

        Return JSON format focusing on user-friendly language:
        {
          "user_friendly_roads": [
            {
              "suggested_id": 0,
              "user_friendly_names": ["main street", "the wide road", "Route 20"],
              "natural_description": "Wide east-west road that runs through the center",
              "characteristics": ["wide", "busy", "main thoroughfare"],
              "visual_landmarks": ["Fuchu Station", "Ozeki supermarket"],
              "relative_position": "central horizontal road",
              "user_would_say": "the main road" 
            }
          ],
          "user_friendly_intersections": [
            {
              "suggested_id": 0,
              "user_friendly_names": ["station intersection", "main crossing", "big intersection"],
              "natural_description": "Major intersection near Fuchu Station with traffic lights",
              "visual_landmarks": ["Fuchu Station North", "Ozeki", "traffic lights"],
              "user_would_say": "the intersection by the station",
              "connecting_roads_description": "where the main road meets the north-south street"
            }
          ],
          "user_friendly_lanes": [
            {
              "road_reference": 0,
              "direction": "eastbound",
              "user_friendly_names": ["right lane heading to station", "eastbound main road lane"],
              "natural_description": "Right lane of main road heading towards station area",
              "landmarks_visible": ["station entrance", "Ozeki on the right"],
              "user_would_say": "the lane going towards the station"
            }
          ],
          "spatial_relationships": [
            {
              "relationship_type": "road_to_road_via_intersection",
              "description": "Main road connects to north road at station intersection",
              "entities": {
                "road_1": {"id": 0, "role": "from", "user_name": "main road"},
                "road_2": {"id": 1, "role": "to", "user_name": "north road"},  
                "intersection": {"id": 0, "role": "connector", "user_name": "station intersection"}
              },
              "navigation_phrase": "from main road, turn right at station intersection to go up north road"
            }
          ],
          "landmark_details": [
            {
              "name": "Fuchu Station North",
              "type": "transportation_hub",
              "visibility": "highly_visible",
              "navigation_value": "primary_reference_point",
              "user_descriptions": ["the station", "Fuchu station", "train station"],
              "relative_to_roads": "main landmark on the central road"
            }
          ]
        }
        """
        
        try:
            print("🧠 Analyzing images with AWS Bedrock Claude for user-friendly aliases...")
            
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 8000,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": roadmap_b64
                                    }
                                },
                                {
                                    "type": "image", 
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": satellite_b64
                                    }
                                }
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            print("\n" + "="*80)
            print("🔍 BEDROCK USER-FRIENDLY ANALYSIS:")
            print("="*80)
            print(analysis_text)
            print("="*80)
            
            # Extract JSON
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = analysis_text[json_start:json_end]
                try:
                    metadata_json = json.loads(json_text)
                    print("✅ Successfully parsed Bedrock user-friendly analysis")
                    return metadata_json
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    return {"error": "JSON parse failed", "raw_response": analysis_text}
            else:
                return {"error": "No JSON found", "raw_response": analysis_text}
                
        except Exception as e:
            print(f"💥 Bedrock analysis failed: {e}")
            return {"error": str(e)}

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

    def is_point_at_edge(self, point):
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

    def build_consistent_road_intersection_mapping(self):
        """Build consistent mapping between roads and intersections"""
        print("\n🔗 BUILDING CONSISTENT ROAD-INTERSECTION MAPPING...")
        print("-" * 60)
        
        # Reset mappings
        self.road_to_intersections = {}
        self.intersection_to_roads = {}
        
        # For each road, find which intersections it connects to
        for road in self.roads:
            road_id = road['id']
            points = road['points']
            connected_intersections = []
            
            for intersection in self.intersections:
                int_id = intersection['id']
                int_center = intersection['center']
                
                # Check if road connects to this intersection
                start_dist = sqrt((points[0][0] - int_center[0])**2 + (points[0][1] - int_center[1])**2)
                end_dist = sqrt((points[-1][0] - int_center[0])**2 + (points[-1][1] - int_center[1])**2)
                
                if start_dist <= INTERSECTION_RADIUS or end_dist <= INTERSECTION_RADIUS:
                    connected_intersections.append({
                        'intersection_id': int_id,
                        'start_connected': start_dist <= INTERSECTION_RADIUS,
                        'end_connected': end_dist <= INTERSECTION_RADIUS,
                        'start_distance': start_dist,
                        'end_distance': end_dist
                    })
            
            self.road_to_intersections[road_id] = connected_intersections
            print(f"   Road {road_id}: connects to {len(connected_intersections)} intersections")
        
        # Build reverse mapping
        for intersection in self.intersections:
            int_id = intersection['id']
            connected_roads = []
            
            for road_id, connections in self.road_to_intersections.items():
                for conn in connections:
                    if conn['intersection_id'] == int_id:
                        road = next(r for r in self.roads if r['id'] == road_id)
                        connected_roads.append({
                            'road_id': road_id,
                            'start_connected': conn['start_connected'],
                            'end_connected': conn['end_connected'],
                            'road_points': road['points']
                        })
                        break
            
            self.intersection_to_roads[int_id] = connected_roads
            print(f"   Intersection {int_id}: connects to {len(connected_roads)} roads")

    def generate_common_language_vocabulary(self):
        """Generate user-friendly aliases and spatial relationships based on Bedrock analysis"""
        print("\n🗣️  GENERATING COMMON LANGUAGE VOCABULARY...")
        print("-" * 60)
        
        bedrock_roads = self.bedrock_metadata.get('user_friendly_roads', [])
        bedrock_intersections = self.bedrock_metadata.get('user_friendly_intersections', [])
        bedrock_lanes = self.bedrock_metadata.get('user_friendly_lanes', [])
        bedrock_relationships = self.bedrock_metadata.get('spatial_relationships', [])
        bedrock_landmarks = self.bedrock_metadata.get('landmark_details', [])
        
        # Generate road aliases
        for road in self.roads:
            road_id = road['id']
            
            # Try to match with Bedrock analysis
            bedrock_road = None
            if road_id < len(bedrock_roads):
                bedrock_road = bedrock_roads[road_id]
            elif bedrock_roads:  # Fallback to first available if ID mismatch
                bedrock_road = bedrock_roads[0]
            
            # Calculate road characteristics for fallback descriptions
            direction_info = self.calculate_road_direction(road['points'])
            curvature = self.calculate_road_curvature(road['points'])
            width_category = self.estimate_road_width_category(road['points'])
            edge_analysis = road.get('metadata', {}).get('edge_analysis', {})
            
            # Generate user-friendly aliases
            aliases = {
                'primary_name': bedrock_road.get('user_would_say', f'Road {road_id}') if bedrock_road else f'Road {road_id}',
                'alternative_names': bedrock_road.get('user_friendly_names', []) if bedrock_road else [f'Road {road_id}', f'the {width_category} road'],
                'natural_description': bedrock_road.get('natural_description', f'{width_category.capitalize()} {direction_info.get("simple", "road")} with {curvature} curvature') if bedrock_road else f'{width_category.capitalize()} {direction_info.get("simple", "road")} with {curvature} curvature',
                'characteristics': bedrock_road.get('characteristics', [width_category, curvature, direction_info.get('simple', 'unknown')]) if bedrock_road else [width_category, curvature, direction_info.get('simple', 'unknown')],
                'visual_landmarks': bedrock_road.get('visual_landmarks', []) if bedrock_road else [],
                'relative_position': bedrock_road.get('relative_position', f'{direction_info.get("cardinal", "unknown")} area') if bedrock_road else f'{direction_info.get("cardinal", "unknown")} area',
                'user_friendly_references': {
                    'formal': f'Road {road_id}',
                    'casual': bedrock_road.get('user_would_say', f'the {width_category} road') if bedrock_road else f'the {width_category} road',
                    'descriptive': f'the {width_category} {direction_info.get("simple", "road")}',
                    'landmark_based': f'road near {bedrock_road.get("visual_landmarks", ["unknown area"])[0]}' if bedrock_road and bedrock_road.get('visual_landmarks') else 'road in unknown area'
                },
                'edge_context': {
                    'is_entry_exit': edge_analysis.get('has_edge_connection', False),
                    'edge_description': f'leads to {"/".join(edge_analysis.get("edge_sides", []))} edge' if edge_analysis.get('has_edge_connection') else 'internal road',
                    'user_navigation': f'road that goes to the {"/".join(edge_analysis.get("edge_sides", []))}' if edge_analysis.get('has_edge_connection') else 'road that stays in the area'
                }
            }
            
            self.common_language_vocabulary['roads'][road_id] = aliases
            print(f"   Road {road_id}: '{aliases['primary_name']}' - {aliases['natural_description']}")
        
        # Generate intersection aliases
        for intersection in self.intersections:
            int_id = intersection['id']
            
            # Try to match with Bedrock analysis
            bedrock_intersection = None
            if int_id < len(bedrock_intersections):
                bedrock_intersection = bedrock_intersections[int_id]
            elif bedrock_intersections:  # Fallback
                bedrock_intersection = bedrock_intersections[0]
            
            # Get connected roads for context
            connected_roads = self.intersection_to_roads.get(int_id, [])
            connected_road_names = []
            for conn in connected_roads:
                road_alias = self.common_language_vocabulary['roads'].get(conn['road_id'], {})
                if road_alias:
                    connected_road_names.append(road_alias['primary_name'])
            
            int_metadata = intersection.get('metadata', {})
            int_type = int_metadata.get('intersection_type', 'unknown')
            
            # Generate intersection aliases
            aliases = {
                'primary_name': bedrock_intersection.get('user_would_say', f'Intersection {int_id}') if bedrock_intersection else f'Intersection {int_id}',
                'alternative_names': bedrock_intersection.get('user_friendly_names', []) if bedrock_intersection else [f'Intersection {int_id}', f'the {int_type}'],
                'natural_description': bedrock_intersection.get('natural_description', f'{int_type.replace("_", " ").title()} connecting {len(connected_roads)} roads') if bedrock_intersection else f'{int_type.replace("_", " ").title()} connecting {len(connected_roads)} roads',
                'visual_landmarks': bedrock_intersection.get('visual_landmarks', int_metadata.get('landmarks', [])) if bedrock_intersection else int_metadata.get('landmarks', []),
                'connecting_roads_description': bedrock_intersection.get('connecting_roads_description', f'where {" and ".join(connected_road_names[:2])} meet') if bedrock_intersection and connected_road_names else f'where {" and ".join(connected_road_names[:2])} meet' if connected_road_names else 'road junction',
                'user_friendly_references': {
                    'formal': f'Intersection {int_id}',
                    'casual': bedrock_intersection.get('user_would_say', f'the {int_type.replace("_", " ")}') if bedrock_intersection else f'the {int_type.replace("_", " ")}',
                    'descriptive': f'the {int_type.replace("_", " ")} with {len(connected_roads)} roads',
                    'landmark_based': f'intersection near {bedrock_intersection.get("visual_landmarks", ["unknown"])[0]}' if bedrock_intersection and bedrock_intersection.get('visual_landmarks') else f'intersection near {int_metadata.get("landmarks", ["unknown area"])[0]}' if int_metadata.get('landmarks') else 'intersection in unknown area'
                },
                'navigation_context': {
                    'turn_options': self.generate_turn_options_description(int_id, connected_roads),
                    'traffic_info': f'has traffic signals' if int_metadata.get('traffic_signals', False) else 'no traffic signals',
                    'wait_time': f'typical wait: {int_metadata.get("estimated_wait_time", 10)} seconds'
                }
            }
            
            self.common_language_vocabulary['intersections'][int_id] = aliases
            print(f"   Intersection {int_id}: '{aliases['primary_name']}' - {aliases['natural_description']}")
        
        # Generate lane aliases
        for road in self.roads:
            road_id = road['id']
            road_alias = self.common_language_vocabulary['roads'].get(road_id, {})
            
            # Generate forward and backward lane aliases
            for direction in ['forward', 'backward']:
                lane_id = f"road_{road_id}_{direction}_trunk"
                
                # Try to match with Bedrock lanes
                bedrock_lane = None
                for bl in bedrock_lanes:
                    if bl.get('road_reference') == road_id and bl.get('direction') == f'{direction}bound':
                        bedrock_lane = bl
                        break
                
                direction_description = direction + 'bound'
                road_name = road_alias.get('primary_name', f'Road {road_id}')
                
                aliases = {
                    'primary_name': bedrock_lane.get('user_would_say', f'{direction} lane of {road_name}') if bedrock_lane else f'{direction} lane of {road_name}',
                    'alternative_names': bedrock_lane.get('user_friendly_names', []) if bedrock_lane else [f'{direction} lane of {road_name}', f'{direction_description} {road_name}'],
                    'natural_description': bedrock_lane.get('natural_description', f'{direction.capitalize()} lane of {road_alias.get("natural_description", road_name)}') if bedrock_lane else f'{direction.capitalize()} lane of {road_alias.get("natural_description", road_name)}',
                    'landmarks_visible': bedrock_lane.get('landmarks_visible', road_alias.get('visual_landmarks', [])) if bedrock_lane else road_alias.get('visual_landmarks', []),
                    'user_friendly_references': {
                        'formal': lane_id,
                        'casual': bedrock_lane.get('user_would_say', f'{direction} lane') if bedrock_lane else f'{direction} lane',
                        'descriptive': f'{direction} lane of the {road_alias.get("characteristics", ["unknown"])[0]} road',
                        'landmark_based': f'{direction} lane {road_alias.get("user_friendly_references", {}).get("landmark_based", "")}'
                    },
                    'navigation_context': {
                        'part_of_road': road_name,
                        'direction_info': direction_description,
                        'connects_to': [conn['intersection_id'] for conn in self.road_to_intersections.get(road_id, [])]
                    }
                }
                
                self.common_language_vocabulary['lanes'][lane_id] = aliases
                print(f"   Lane {lane_id}: '{aliases['primary_name']}'")
        
        # Generate spatial relationships
        print(f"\n🔄 GENERATING SPATIAL RELATIONSHIPS...")
        
        spatial_relationships = []
        
        # Add Bedrock relationships
        for relationship in bedrock_relationships:
            spatial_relationships.append({
                'relationship_type': relationship.get('relationship_type', 'unknown'),
                'description': relationship.get('description', 'unknown connection'),
                'navigation_phrase': relationship.get('navigation_phrase', ''),
                'entities': relationship.get('entities', {}),
                'user_friendly': True,
                'source': 'bedrock_analysis'
            })
        
        # Generate relationships between all roads and intersections
        for road_id, intersections_list in self.road_to_intersections.items():
            road_alias = self.common_language_vocabulary['roads'].get(road_id, {})
            road_name = road_alias.get('primary_name', f'Road {road_id}')
            
            for int_conn in intersections_list:
                int_id = int_conn['intersection_id']
                int_alias = self.common_language_vocabulary['intersections'].get(int_id, {})
                int_name = int_alias.get('primary_name', f'Intersection {int_id}')
                
                # Road to intersection relationship
                spatial_relationships.append({
                    'relationship_type': 'road_connects_to_intersection',
                    'description': f'{road_name} connects to {int_name}',
                    'navigation_phrase': f'take {road_name} to reach {int_name}',
                    'entities': {
                        'road': {'id': road_id, 'name': road_name, 'role': 'origin'},
                        'intersection': {'id': int_id, 'name': int_name, 'role': 'destination'}
                    },
                    'user_friendly': True,
                    'source': 'generated_from_structure'
                })
        
        # Generate road-to-road relationships via intersections
        for int_id, roads_list in self.intersection_to_roads.items():
            int_alias = self.common_language_vocabulary['intersections'].get(int_id, {})
            int_name = int_alias.get('primary_name', f'Intersection {int_id}')
            
            # Create relationships between each pair of roads at this intersection
            for i, road_conn_1 in enumerate(roads_list):
                for road_conn_2 in roads_list[i+1:]:
                    road_1_id = road_conn_1['road_id']
                    road_2_id = road_conn_2['road_id']
                    
                    road_1_alias = self.common_language_vocabulary['roads'].get(road_1_id, {})
                    road_2_alias = self.common_language_vocabulary['roads'].get(road_2_id, {})
                    
                    road_1_name = road_1_alias.get('primary_name', f'Road {road_1_id}')
                    road_2_name = road_2_alias.get('primary_name', f'Road {road_2_id}')
                    
                    # Calculate turn direction
                    road_1 = next(r for r in self.roads if r['id'] == road_1_id)
                    road_2 = next(r for r in self.roads if r['id'] == road_2_id)
                    intersection = next(i for i in self.intersections if i['id'] == int_id)
                    turn_type = self.calculate_turn_type(road_1, road_2, intersection['center'])
                    
                    spatial_relationships.append({
                        'relationship_type': 'road_to_road_via_intersection',
                        'description': f'{road_1_name} connects to {road_2_name} via {int_name}',
                        'navigation_phrase': f'from {road_1_name}, {self.turn_type_to_phrase(turn_type)} at {int_name} to reach {road_2_name}',
                        'turn_type': turn_type,
                        'entities': {
                            'road_from': {'id': road_1_id, 'name': road_1_name, 'role': 'origin'},
                            'road_to': {'id': road_2_id, 'name': road_2_name, 'role': 'destination'},
                            'intersection': {'id': int_id, 'name': int_name, 'role': 'connector'}
                        },
                        'user_friendly': True,
                        'source': 'generated_from_structure'
                    })
        
        self.common_language_vocabulary['spatial_relationships'] = spatial_relationships
        
        # Process landmark details
        landmarks_dict = {}
        for landmark in bedrock_landmarks:
            landmark_name = landmark.get('name', 'Unknown Landmark')
            landmarks_dict[landmark_name] = {
                'type': landmark.get('type', 'unknown'),
                'visibility': landmark.get('visibility', 'medium'),
                'navigation_value': landmark.get('navigation_value', 'medium'),
                'user_descriptions': landmark.get('user_descriptions', [landmark_name]),
                'relative_to_roads': landmark.get('relative_to_roads', 'unknown location'),
                'primary_reference': landmark.get('user_descriptions', [landmark_name])[0] if landmark.get('user_descriptions') else landmark_name
            }
        
        self.common_language_vocabulary['landmarks'] = landmarks_dict
        
        print(f"✅ Generated {len(self.common_language_vocabulary['roads'])} road aliases")
        print(f"✅ Generated {len(self.common_language_vocabulary['intersections'])} intersection aliases")
        print(f"✅ Generated {len(self.common_language_vocabulary['lanes'])} lane aliases")
        print(f"✅ Generated {len(spatial_relationships)} spatial relationships")
        print(f"✅ Processed {len(landmarks_dict)} landmark details")

    def turn_type_to_phrase(self, turn_type):
        """Convert turn type to natural language phrase"""
        phrases = {
            'straight': 'continue straight',
            'left': 'turn left',
            'right': 'turn right',
            'u_turn': 'make a U-turn',
            'slight_left': 'turn slightly left',
            'slight_right': 'turn slightly right',
            'unknown': 'continue'
        }
        return phrases.get(turn_type, 'continue')

    def generate_turn_options_description(self, intersection_id, connected_roads):
        """Generate description of turn options at intersection"""
        if len(connected_roads) <= 1:
            return "dead end - no turns available"
        elif len(connected_roads) == 2:
            return "straight through or turn around"
        elif len(connected_roads) == 3:
            return "can turn left, right, or continue straight"
        elif len(connected_roads) == 4:
            return "can turn left, right, continue straight, or U-turn"
        else:
            return f"complex intersection with {len(connected_roads)} directions"

    def enhance_with_integrated_metadata(self):
        """Enhance roads and intersections with consistent, integrated metadata including user-friendly aliases"""
        print("\n🎯 ENHANCING WITH INTEGRATED METADATA (including user-friendly aliases)...")
        print("-" * 70)
        
        # Process edge analysis first
        edge_roads_count = 0
        roads_with_edges = []
        
        for road in self.roads:
            road_id = road['id']
            points = road['points']
            
            start_edge_info = self.is_point_at_edge(points[0])
            end_edge_info = self.is_point_at_edge(points[-1])
            
            has_edge_connection = start_edge_info['is_edge'] or end_edge_info['is_edge']
            
            if has_edge_connection:
                edge_roads_count += 1
                roads_with_edges.append((road_id, start_edge_info, end_edge_info))
        
        # Generate consistent edge IDs
        self.generate_edge_ids(roads_with_edges)
        
        # Enhance roads with comprehensive metadata including user-friendly aliases
        for road in self.roads:
            road_id = road['id']
            points = road['points']
            
            # Find edge info for this road
            road_edge_data = next((data for data in roads_with_edges if data[0] == road_id), None)
            start_edge_info = road_edge_data[1] if road_edge_data else {'is_edge': False}
            end_edge_info = road_edge_data[2] if road_edge_data else {'is_edge': False}
            
            # Get user-friendly aliases
            road_aliases = self.common_language_vocabulary['roads'].get(road_id, {})
            
            # Try to match with Bedrock analysis
            bedrock_road_info = self.match_road_with_bedrock(road_id, points)
            
            # Calculate comprehensive metadata
            direction_info = self.calculate_road_direction(points)
            curvature = self.calculate_road_curvature(points)
            width_category = self.estimate_road_width_category(points)
            
            road['metadata'] = {
                # Basic properties
                'name': road_aliases.get('primary_name', bedrock_road_info.get('name', f'Road_{road_id}')),
                'alt_names': road_aliases.get('alternative_names', bedrock_road_info.get('alt_names', [])),
                'road_class': bedrock_road_info.get('class', 'local_street' if width_category == 'narrow' else 'main_road'),
                'road_type': bedrock_road_info.get('type', 'local'),
                'estimated_speed_limit': bedrock_road_info.get('speed_limit', 30 if width_category == 'narrow' else 40),
                'traffic_density': 'low' if width_category == 'narrow' else 'medium',
                'curvature': curvature,
                'width_category': width_category,
                'landmarks': bedrock_road_info.get('landmarks', road_aliases.get('visual_landmarks', [])),
                'estimated_length_meters': len(points) * 0.5,
                'priority': 1 if width_category == 'narrow' else 2 if width_category == 'medium' else 3,
                
                # USER-FRIENDLY ALIASES - This is the key addition from the PDF
                'user_friendly_aliases': road_aliases,
                'common_language': {
                    'primary_user_name': road_aliases.get('primary_name', f'Road {road_id}'),
                    'natural_description': road_aliases.get('natural_description', f'{width_category} road'),
                    'user_references': road_aliases.get('user_friendly_references', {}),
                    'characteristics_user_sees': road_aliases.get('characteristics', []),
                    'relative_position_description': road_aliases.get('relative_position', 'unknown area'),
                    'edge_context_for_user': road_aliases.get('edge_context', {})
                },
                
                # Direction information
                'direction': direction_info,
                'cardinal_direction': direction_info.get('cardinal', 'unknown') if isinstance(direction_info, dict) else 'unknown',
                'simple_direction': direction_info.get('simple', 'unknown') if isinstance(direction_info, dict) else str(direction_info),
                
                # Edge analysis
                'edge_analysis': {
                    'has_edge_connection': start_edge_info.get('is_edge', False) or end_edge_info.get('is_edge', False),
                    'edge_sides': list(set(start_edge_info.get('edge_sides', []) + end_edge_info.get('edge_sides', []))),
                    'start_edge': start_edge_info,
                    'end_edge': end_edge_info,
                    'true_start_directions': start_edge_info.get('edge_sides', []) if start_edge_info.get('is_edge') else [],
                    'true_end_directions': end_edge_info.get('edge_sides', []) if end_edge_info.get('is_edge') else [],
                    'is_geographic_entry_point': start_edge_info.get('is_edge', False),
                    'is_geographic_exit_point': end_edge_info.get('is_edge', False)
                },
                
                # Navigation properties
                'can_turn_left': True,
                'can_turn_right': True,
                'can_go_straight': True,
                'parking_available': width_category == 'wide',
                
                # CONSISTENT intersection connections
                'connects_to_intersections': [conn['intersection_id'] for conn in self.road_to_intersections.get(road_id, [])],
                'intersection_connections': self.road_to_intersections.get(road_id, []),
                
                # NARRATIVE RELATIONSHIPS - New addition for semantic understanding
                'narrative_relationships': {
                    'connects_to_roads': self.get_connected_roads_narrative(road_id),
                    'spatial_description': self.get_road_spatial_narrative(road_id),
                    'landmark_relationships': self.get_road_landmark_relationships(road_id),
                    'navigation_context': self.get_road_navigation_context(road_id)
                }
            }
            
            # Add to geographic mapping if has edge connection
            if road['metadata']['edge_analysis']['has_edge_connection']:
                for side in road['metadata']['edge_analysis']['edge_sides']:
                    if side in self.geographic_road_map:
                        self.geographic_road_map[side].append(road_id)
        
        # Enhance intersections with consistent metadata including user-friendly aliases
        for intersection in self.intersections:
            int_id = intersection['id']
            center = intersection['center']
            
            # Get connected roads with consistent IDs
            connected_roads = self.intersection_to_roads.get(int_id, [])
            
            # Get user-friendly aliases
            int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
            
            # Try to match with Bedrock analysis
            bedrock_int_info = self.match_intersection_with_bedrock(int_id, center, connected_roads)
            
            # Calculate intersection type
            int_type = self.determine_intersection_type(len(connected_roads))
            
            # Check if intersection is at edge
            intersection_edge_info = self.is_point_at_edge(center)
            
            intersection['metadata'] = {
                # Basic properties
                'intersection_type': int_type,
                'connected_roads_count': len(connected_roads),
                'connected_road_ids': [road['road_id'] for road in connected_roads],
                
                # Bedrock-enhanced info
                'landmarks': bedrock_int_info.get('landmarks', int_aliases.get('visual_landmarks', [])),
                'nearby_businesses': bedrock_int_info.get('businesses', []),
                'traffic_signals': bedrock_int_info.get('traffic_signals', len(connected_roads) >= 3),
                'estimated_wait_time': bedrock_int_info.get('wait_time', 15 if len(connected_roads) >= 3 else 5),
                
                # USER-FRIENDLY ALIASES
                'user_friendly_aliases': int_aliases,
                'common_language': {
                    'primary_user_name': int_aliases.get('primary_name', f'Intersection {int_id}'),
                    'natural_description': int_aliases.get('natural_description', f'{int_type.replace("_", " ")}'),
                    'user_references': int_aliases.get('user_friendly_references', {}),
                    'connecting_roads_description': int_aliases.get('connecting_roads_description', 'road junction'),
                    'navigation_context': int_aliases.get('navigation_context', {})
                },
                
                # Edge analysis
                'edge_analysis': {
                    'intersection_edge': intersection_edge_info,
                    'is_edge_intersection': intersection_edge_info.get('is_edge', False),
                    'edge_sides': intersection_edge_info.get('edge_sides', [])
                },
                
                # Navigation aids
                'can_turn_left': len(connected_roads) >= 3,
                'can_turn_right': len(connected_roads) >= 3,
                'can_go_straight': len(connected_roads) >= 2,
                
                # CONSISTENT road connections
                'road_connections': connected_roads,
                
                # NARRATIVE RELATIONSHIPS
                'narrative_relationships': {
                    'connecting_roads_narrative': self.get_intersection_roads_narrative(int_id),
                    'landmark_context': self.get_intersection_landmark_context(int_id),
                    'turn_options_narrative': self.get_intersection_turn_narrative(int_id),
                    'spatial_position': self.get_intersection_spatial_position(int_id, center)
                }
            }
        
        print(f"✅ Enhanced {len(self.roads)} roads and {len(self.intersections)} intersections with user-friendly aliases")
        print(f"🎯 Found {edge_roads_count} roads with edge connections")
        
        # Print geographic distribution
        for direction, road_ids in self.geographic_road_map.items():
            if road_ids:
                road_names = [self.common_language_vocabulary['roads'].get(rid, {}).get('primary_name', f'Road {rid}') for rid in road_ids]
                print(f"   {direction.upper()}: {len(road_ids)} roads - {', '.join(road_names[:2])}")

    # NEW METHODS for narrative relationships
    def get_connected_roads_narrative(self, road_id):
        """Get narrative description of connected roads"""
        connected_ints = self.road_to_intersections.get(road_id, [])
        narrative = []
        
        for int_conn in connected_ints:
            int_id = int_conn['intersection_id']
            other_roads = self.intersection_to_roads.get(int_id, [])
            
            int_alias = self.common_language_vocabulary['intersections'].get(int_id, {})
            int_name = int_alias.get('primary_name', f'Intersection {int_id}')
            
            connected_road_names = []
            for other_road in other_roads:
                if other_road['road_id'] != road_id:
                    other_road_alias = self.common_language_vocabulary['roads'].get(other_road['road_id'], {})
                    connected_road_names.append(other_road_alias.get('primary_name', f'Road {other_road["road_id"]}'))
            
            if connected_road_names:
                narrative.append({
                    'intersection': int_name,
                    'connected_roads': connected_road_names,
                    'description': f'connects to {", ".join(connected_road_names)} at {int_name}'
                })
        
        return narrative

    def get_road_spatial_narrative(self, road_id):
        """Get spatial narrative description for road"""
        road_aliases = self.common_language_vocabulary['roads'].get(road_id, {})
        road = next((r for r in self.roads if r['id'] == road_id), None)
        
        if not road:
            return "unknown spatial position"
        
        edge_analysis = road.get('metadata', {}).get('edge_analysis', {})
        direction_info = self.calculate_road_direction(road['points'])
        
        spatial_desc = []
        
        # Add directional info
        spatial_desc.append(f"runs {direction_info.get('simple', 'unknown direction')}")
        
        # Add edge info
        if edge_analysis.get('has_edge_connection', False):
            edge_sides = edge_analysis.get('edge_sides', [])
            if edge_analysis.get('is_geographic_entry_point'):
                spatial_desc.append(f"enters from {'/'.join(edge_sides)} edge")
            if edge_analysis.get('is_geographic_exit_point'):
                spatial_desc.append(f"exits to {'/'.join(edge_sides)} edge")
        else:
            spatial_desc.append("stays within the area")
        
        # Add position relative to landmarks
        landmarks = road_aliases.get('visual_landmarks', [])
        if landmarks:
            spatial_desc.append(f"passes {landmarks[0]}")
        
        return " • ".join(spatial_desc) if spatial_desc else "unknown spatial context"

    def get_road_landmark_relationships(self, road_id):
        """Get landmark relationships for road"""
        road_aliases = self.common_language_vocabulary['roads'].get(road_id, {})
        landmarks = road_aliases.get('visual_landmarks', [])
        
        relationships = []
        for landmark in landmarks:
            landmark_details = self.common_language_vocabulary['landmarks'].get(landmark, {})
            relationships.append({
                'landmark': landmark,
                'type': landmark_details.get('type', 'unknown'),
                'relationship': f'road passes {landmark}',
                'user_description': landmark_details.get('primary_reference', landmark),
                'navigation_value': landmark_details.get('navigation_value', 'medium')
            })
        
        return relationships

    def get_road_navigation_context(self, road_id):
        """Get navigation context for road"""
        road_aliases = self.common_language_vocabulary['roads'].get(road_id, {})
        road = next((r for r in self.roads if r['id'] == road_id), None)
        
        if not road:
            return {}
        
        connected_ints = self.road_to_intersections.get(road_id, [])
        
        context = {
            'primary_name': road_aliases.get('primary_name', f'Road {road_id}'),
            'how_user_refers': road_aliases.get('user_friendly_references', {}).get('casual', f'Road {road_id}'),
            'navigation_landmarks': road_aliases.get('landmarks_visible', []),
            'connects_to': len(connected_ints),
            'turn_opportunities': [self.common_language_vocabulary['intersections'].get(conn['intersection_id'], {}).get('primary_name', f'Intersection {conn["intersection_id"]}') for conn in connected_ints],
            'user_navigation_phrase': f'take {road_aliases.get("primary_name", f"Road {road_id}")} to reach your destination'
        }
        
        return context

    def get_intersection_roads_narrative(self, int_id):
        """Get narrative of roads connecting to intersection"""
        connected_roads = self.intersection_to_roads.get(int_id, [])
        narrative = []
        
        for road_conn in connected_roads:
            road_id = road_conn['road_id']
            road_alias = self.common_language_vocabulary['roads'].get(road_id, {})
            road_name = road_alias.get('primary_name', f'Road {road_id}')
            road_description = road_alias.get('natural_description', 'unknown road')
            
            narrative.append({
                'road_id': road_id,
                'road_name': road_name,
                'description': road_description,
                'connection_point': 'start' if road_conn.get('start_connected', False) else 'end' if road_conn.get('end_connected', False) else 'middle',
                'user_phrase': f'{road_name} {road_conn.get("connection_point", "connects here")}'
            })
        
        return narrative

    def get_intersection_landmark_context(self, int_id):
        """Get landmark context for intersection"""
        int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
        landmarks = int_aliases.get('visual_landmarks', [])
        
        context = []
        for landmark in landmarks:
            landmark_details = self.common_language_vocabulary['landmarks'].get(landmark, {})
            context.append({
                'landmark': landmark,
                'user_reference': landmark_details.get('primary_reference', landmark),
                'type': landmark_details.get('type', 'unknown'),
                'visibility': landmark_details.get('visibility', 'medium'),
                'navigation_phrase': f'intersection near {landmark_details.get("primary_reference", landmark)}'
            })
        
        return context

    def get_intersection_turn_narrative(self, int_id):
        """Get turn options narrative for intersection"""
        int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
        connected_roads = self.intersection_to_roads.get(int_id, [])
        
        if len(connected_roads) <= 1:
            return "dead end - cannot continue"
        
        turn_options = []
        intersection = next((i for i in self.intersections if i['id'] == int_id), None)
        
        if not intersection:
            return []
        
        # Generate turn descriptions for each pair of roads
        for i, from_road in enumerate(connected_roads):
            for to_road in connected_roads[i+1:]:
                from_road_id = from_road['road_id']
                to_road_id = to_road['road_id']
                
                from_road_alias = self.common_language_vocabulary['roads'].get(from_road_id, {})
                to_road_alias = self.common_language_vocabulary['roads'].get(to_road_id, {})
                
                from_road_name = from_road_alias.get('primary_name', f'Road {from_road_id}')
                to_road_name = to_road_alias.get('primary_name', f'Road {to_road_id}')
                
                from_road_obj = next(r for r in self.roads if r['id'] == from_road_id)
                to_road_obj = next(r for r in self.roads if r['id'] == to_road_id)
                
                turn_type = self.calculate_turn_type(from_road_obj, to_road_obj, intersection['center'])
                turn_phrase = self.turn_type_to_phrase(turn_type)
                
                turn_options.append({
                    'from_road': from_road_name,
                    'to_road': to_road_name,
                    'turn_type': turn_type,
                    'user_instruction': f'from {from_road_name}, {turn_phrase} to reach {to_road_name}',
                    'navigation_phrase': f'{turn_phrase} onto {to_road_name}'
                })
        
        return turn_options

    def get_intersection_spatial_position(self, int_id, center):
        """Get spatial position description for intersection"""
        # Determine position relative to canvas
        x, y = center
        width, height = CANVAS_SIZE
        
        horizontal_pos = 'left' if x < width/3 else 'right' if x > 2*width/3 else 'center'
        vertical_pos = 'top' if y < height/3 else 'bottom' if y > 2*height/3 else 'middle'
        
        # Check if near edge
        edge_info = self.is_point_at_edge(center)
        edge_context = ""
        if edge_info['is_edge']:
            edge_sides = '/'.join(edge_info['edge_sides'])
            edge_context = f" (near {edge_sides} edge)"
        
        position_desc = f"{vertical_pos}-{horizontal_pos} of the area{edge_context}"
        
        return {
            'general_position': position_desc,
            'coordinates': {'x': x, 'y': y},
            'relative_position': f'{horizontal_pos} side, {vertical_pos} area',
            'edge_context': edge_context,
            'user_spatial_reference': f'intersection in the {vertical_pos} {horizontal_pos} area{edge_context}'
        }

    def generate_edge_ids(self, roads_with_edges):
        """Generate consistent edge IDs"""
        edge_counters = {'west': 0, 'east': 0, 'north': 0, 'south': 0}
        
        for direction in ['west', 'east', 'north', 'south']:
            direction_roads = []
            
            for road_data in roads_with_edges:
                road_id, start_edge, end_edge = road_data
                
                if (start_edge.get('is_edge') and direction in start_edge.get('edge_sides', [])) or \
                   (end_edge.get('is_edge') and direction in end_edge.get('edge_sides', [])):
                    
                    if start_edge.get('is_edge') and direction in start_edge.get('edge_sides', []):
                        coord_info = start_edge['edge_coordinates'][direction]
                    else:
                        coord_info = end_edge['edge_coordinates'][direction]
                    
                    sort_key = coord_info['y'] if direction in ['west', 'east'] else coord_info['x']
                    direction_roads.append((road_data, sort_key))
            
            direction_roads.sort(key=lambda x: x[1])
            
            for i, (road_data, _) in enumerate(direction_roads):
                road_id, start_edge, end_edge = road_data
                edge_id = f"{direction}_end_{i:02d}"
                
                if start_edge.get('is_edge') and direction in start_edge.get('edge_sides', []):
                    start_edge['edge_id'] = edge_id
                    # Use user-friendly name if available
                    road_alias = self.common_language_vocabulary['roads'].get(road_id, {})
                    road_name = road_alias.get('primary_name', f'Road_{road_id}')
                    
                    self.edge_entry_points[edge_id] = {
                        'road_id': road_id,
                        'road_name': road_name,
                        'user_friendly_name': road_alias.get('primary_name', f'Road_{road_id}'),
                        'point_type': 'start',
                        'directions': start_edge.get('edge_sides', []),
                        'coordinates': start_edge.get('edge_coordinates', {}),
                        'navigation_reference': f'entry point from {direction} via {road_name}'
                    }
                
                if end_edge.get('is_edge') and direction in end_edge.get('edge_sides', []):
                    end_edge['edge_id'] = edge_id
                    road_alias = self.common_language_vocabulary['roads'].get(road_id, {})
                    road_name = road_alias.get('primary_name', f'Road_{road_id}')
                    
                    self.edge_entry_points[edge_id] = {
                        'road_id': road_id,
                        'road_name': road_name,
                        'user_friendly_name': road_alias.get('primary_name', f'Road_{road_id}'),
                        'point_type': 'end', 
                        'directions': end_edge.get('edge_sides', []),
                        'coordinates': end_edge.get('edge_coordinates', {}),
                        'navigation_reference': f'exit point to {direction} via {road_name}'
                    }

    def match_road_with_bedrock(self, road_id, points):
        """Match road with Bedrock analysis"""
        bedrock_roads = self.bedrock_metadata.get('user_friendly_roads', [])
        
        if road_id < len(bedrock_roads):
            bedrock_road = bedrock_roads[road_id]
            return {
                'name': bedrock_road.get('user_would_say', f'Road_{road_id}'),
                'alt_names': bedrock_road.get('user_friendly_names', []),
                'class': bedrock_road.get('characteristics', ['local'])[0] if bedrock_road.get('characteristics') else 'local_street',
                'landmarks': bedrock_road.get('visual_landmarks', []),
                'type': 'main' if 'main' in bedrock_road.get('characteristics', []) else 'local'
            }
        
        return {}

    def match_intersection_with_bedrock(self, int_id, center, connected_roads):
        """Match intersection with Bedrock analysis"""
        bedrock_intersections = self.bedrock_metadata.get('user_friendly_intersections', [])
        
        if int_id < len(bedrock_intersections):
            bedrock_int = bedrock_intersections[int_id]
            return {
                'landmarks': bedrock_int.get('visual_landmarks', []),
                'businesses': bedrock_int.get('visual_landmarks', []),  # Assuming landmarks include businesses
                'traffic_signals': True,  # Default assumption for major intersections
                'wait_time': 15
            }
        
        return {}

    def calculate_road_direction(self, points):
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

    def calculate_road_curvature(self, points):
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

    def estimate_road_width_category(self, points):
        """Estimate road width category"""
        if len(points) < 50:
            return "narrow"
        elif len(points) < 150:
            return "medium"
        else:
            return "wide"

    def determine_intersection_type(self, roads_count):
        """Determine intersection type"""
        if roads_count == 2:
            return "dead_end"
        elif roads_count == 3:
            return "T-intersection"
        elif roads_count == 4:
            return "4-way_cross"
        else:
            return "complex_intersection"

    def generate_integrated_lane_trees(self):
        """Generate lane trees with consistent IDs and comprehensive connections including user-friendly aliases"""
        print("\n🛤️  GENERATING INTEGRATED LANE TREES WITH USER-FRIENDLY ALIASES...")
        print("-" * 70)
        
        self.lane_trees = []
        
        for road in self.roads:
            road_id = road['id']
            points = road['points']
            metadata = road['metadata']
            
            # Create forward and backward lanes
            dual_lanes = self.create_dual_lanes_for_road(road)
            
            # Generate forward lane tree with user-friendly aliases
            forward_tree = self.create_lane_tree(
                road_id, 'forward', dual_lanes['forward_lane'], metadata
            )
            self.lane_trees.append(forward_tree)
            
            # Generate backward lane tree with user-friendly aliases
            backward_tree = self.create_lane_tree(
                road_id, 'backward', dual_lanes['backward_lane'][::-1], metadata
            )
            self.lane_trees.append(backward_tree)
            
            road_name = metadata.get('common_language', {}).get('primary_user_name', f'Road_{road_id}')
            edge_info = ""
            if metadata['edge_analysis']['has_edge_connection']:
                edge_sides = '/'.join(metadata['edge_analysis']['edge_sides'])
                edge_info = f" [EDGE: {edge_sides}]"
            
            print(f"✅ Generated lane trees for {road_name}{edge_info}")
        
        print(f"🎯 Generated {len(self.lane_trees)} integrated lane trees with user-friendly aliases")

    def create_dual_lanes_for_road(self, road):
        """Create offset lanes for road"""
        points = road['points']
        
        try:
            line = LineString(points)
            forward_line = line.parallel_offset(LANE_OFFSET_PX, side="left", join_style=2)
            backward_line = line.parallel_offset(LANE_OFFSET_PX, side="right", join_style=2)
            
            if forward_line.geom_type == "MultiLineString":
                forward_line = max(forward_line.geoms, key=lambda l: l.length)
            if backward_line.geom_type == "MultiLineString":
                backward_line = max(backward_line.geoms, key=lambda l: l.length)
            
            forward_coords = list(forward_line.coords) if not forward_line.is_empty else points
            backward_coords = list(backward_line.coords) if not backward_line.is_empty else points
            
            return {
                'forward_lane': [[float(x), float(y)] for x, y in forward_coords],
                'backward_lane': [[float(x), float(y)] for x, y in backward_coords]
            }
        
        except Exception:
            return {
                'forward_lane': points,
                'backward_lane': points
            }

    def create_lane_tree(self, road_id, direction, lane_points, road_metadata):
        """Create comprehensive lane tree with consistent IDs, connections, and user-friendly aliases"""
        lane_id = f"road_{road_id}_{direction}_trunk"
        
        # Get user-friendly lane aliases
        lane_aliases = self.common_language_vocabulary['lanes'].get(lane_id, {})
        
        # Find branches (connections to other roads)
        branches = self.find_branches_for_lane_tree(road_id, direction, lane_points)
        
        # Find intersection metadata
        intersection_metadata = self.get_intersection_metadata_for_lane(road_id, lane_points)
        
        lane_tree = {
            'lane_id': lane_id,
            'road_id': road_id,
            'direction': direction,
            'lane_type': 'trunk',
            'points': lane_points,
            'branches': branches,
            
            # Road metadata
            'metadata': road_metadata.copy(),
            
            # USER-FRIENDLY LANE ALIASES
            'user_friendly_aliases': lane_aliases,
            'common_language': {
                'primary_user_name': lane_aliases.get('primary_name', f'{direction} lane of Road {road_id}'),
                'natural_description': lane_aliases.get('natural_description', f'{direction} lane'),
                'user_references': lane_aliases.get('user_friendly_references', {}),
                'landmarks_visible': lane_aliases.get('landmarks_visible', []),
                'navigation_context': lane_aliases.get('navigation_context', {})
            },
            
            # NARRATIVE RELATIONSHIPS FOR LANE
            'lane_narrative_relationships': {
                'part_of_road': {
                    'road_id': road_id,
                    'road_name': road_metadata.get('common_language', {}).get('primary_user_name', f'Road {road_id}'),
                    'relationship': f'this lane is part of {road_metadata.get("common_language", {}).get("primary_user_name", f"Road {road_id}")}'
                },
                'direction_context': {
                    'direction': direction,
                    'description': f'{direction}bound lane',
                    'user_phrase': f'the lane going {direction}'
                },
                'landmark_context': self.get_lane_landmark_context(lane_id, road_metadata),
                'connection_narrative': self.get_lane_connection_narrative(lane_id, road_id, branches)
            },
            
            # Intersection metadata (if lane ends at intersection)
            'intersection_metadata': intersection_metadata
        }
        
        # Add lane to road mapping
        self.lane_to_road[lane_id] = road_id
        
        return lane_tree

    def get_lane_landmark_context(self, lane_id, road_metadata):
        """Get landmark context specific to lane"""
        landmarks = road_metadata.get('landmarks', [])
        lane_aliases = self.common_language_vocabulary['lanes'].get(lane_id, {})
        visible_landmarks = lane_aliases.get('landmarks_visible', landmarks)
        
        context = []
        for landmark in visible_landmarks:
            landmark_details = self.common_language_vocabulary['landmarks'].get(landmark, {})
            context.append({
                'landmark': landmark,
                'user_reference': landmark_details.get('primary_reference', landmark),
                'visibility_from_lane': 'visible while driving in this lane',
                'navigation_value': landmark_details.get('navigation_value', 'medium'),
                'user_phrase': f'you can see {landmark_details.get("primary_reference", landmark)} from this lane'
            })
        
        return context

    def get_lane_connection_narrative(self, lane_id, road_id, branches):
        """Get narrative of how this lane connects to other roads"""
        if not branches:
            return "this lane doesn't connect to other roads directly"
        
        connections = []
        for branch in branches:
            target_road_id = branch.get('target_road_id')
            target_road_alias = self.common_language_vocabulary['roads'].get(target_road_id, {})
            target_name = target_road_alias.get('primary_name', f'Road {target_road_id}')
            turn_type = branch.get('turn_type', 'unknown')
            
            connections.append({
                'target_road': target_name,
                'turn_type': turn_type,
                'navigation_instruction': branch.get('navigation_instructions', ''),
                'user_phrase': f'from this lane, {self.turn_type_to_phrase(turn_type)} to reach {target_name}'
            })
        
        return connections

    def find_branches_for_lane_tree(self, road_id, direction, lane_points):
        """Find branches with CONSISTENT IDs, proper route connections, and user-friendly names"""
        branches = []
        
        if not lane_points:
            return branches
        
        # Get the end point of this lane
        lane_end = lane_points[-1]
        
        # Find intersections this road connects to
        road_intersections = self.road_to_intersections.get(road_id, [])
        
        for road_int_conn in road_intersections:
            int_id = road_int_conn['intersection_id']
            intersection = next((i for i in self.intersections if i['id'] == int_id), None)
            
            if not intersection:
                continue
            
            int_center = intersection['center']
            
            # Check if this lane direction connects to this intersection
            lane_to_int_dist = sqrt((lane_end[0] - int_center[0])**2 + (lane_end[1] - int_center[1])**2)
            
            if lane_to_int_dist > INTERSECTION_RADIUS:
                continue
            
            # Find other roads at this intersection
            other_roads = self.intersection_to_roads.get(int_id, [])
            
            for other_road_conn in other_roads:
                other_road_id = other_road_conn['road_id']
                
                if other_road_id == road_id:
                    continue
                
                # Get the other road
                other_road = next((r for r in self.roads if r['id'] == other_road_id), None)
                if not other_road:
                    continue
                
                # Create branch to the other road with user-friendly info
                branch = self.create_branch_to_road(
                    road_id, direction, other_road_id, int_center, intersection
                )
                
                if branch:
                    branches.append(branch)
        
        return branches

    def create_branch_to_road(self, from_road_id, from_direction, to_road_id, int_center, intersection):
        """Create branch connection between roads with comprehensive metadata and user-friendly names"""
        
        # Get target road
        target_road = next((r for r in self.roads if r['id'] == to_road_id), None)
        if not target_road:
            return None
        
        # Create target lane for the branch
        target_dual_lanes = self.create_dual_lanes_for_road(target_road)
        
        # Determine which direction of target road to connect to
        target_lane_points = target_dual_lanes['forward_lane']
        
        # Calculate turn type
        from_road = next((r for r in self.roads if r['id'] == from_road_id), None)
        turn_type = self.calculate_turn_type(from_road, target_road, int_center)
        
        # Get intersection and target road metadata with user-friendly names
        int_metadata = intersection.get('metadata', {})
        target_metadata = target_road.get('metadata', {})
        
        # Get user-friendly names
        int_aliases = self.common_language_vocabulary['intersections'].get(intersection['id'], {})
        target_road_aliases = self.common_language_vocabulary['roads'].get(to_road_id, {})
        
        from_road_aliases = self.common_language_vocabulary['roads'].get(from_road_id, {})
        
        branch = {
            'branch_id': f"road_{from_road_id}_{from_direction}_to_road_{to_road_id}_{turn_type}",
            'turn_type': turn_type,
            'target_road_id': to_road_id,
            'target_lane': target_lane_points,
            
            # Enhanced metadata with USER-FRIENDLY NAMES
            'target_road_name': target_road_aliases.get('primary_name', f'Road_{to_road_id}'),
            'target_road_user_description': target_road_aliases.get('natural_description', f'Road {to_road_id}'),
            'target_road_class': target_metadata.get('road_class', 'unknown'),
            'target_landmarks': target_road_aliases.get('visual_landmarks', []),
            'target_speed_limit': target_metadata.get('estimated_speed_limit', 30),
            'target_traffic_density': target_metadata.get('traffic_density', 'medium'),
            
            # Intersection context with user-friendly names
            'intersection_name': int_aliases.get('primary_name', f'Intersection {intersection["id"]}'),
            'intersection_user_description': int_aliases.get('natural_description', int_metadata.get('intersection_type', 'intersection')),
            'intersection_landmarks': int_aliases.get('visual_landmarks', []),
            'intersection_businesses': int_metadata.get('nearby_businesses', []),
            'intersection_has_signals': int_metadata.get('traffic_signals', False),
            'intersection_type': int_metadata.get('intersection_type', 'unknown'),
            'estimated_wait_time': int_metadata.get('estimated_wait_time', 15),
            
            # Edge information for target with user context
            'target_has_edge_connection': target_metadata.get('edge_analysis', {}).get('has_edge_connection', False),
            'target_edge_sides': target_metadata.get('edge_analysis', {}).get('edge_sides', []),
            'target_edge_context': target_road_aliases.get('edge_context', {}),
            
            # Navigation aids with user-friendly language
            'turn_difficulty': 'easy' if turn_type == 'straight' else 'medium' if turn_type in ['left', 'right'] else 'hard',
            'navigation_instructions': self.generate_navigation_instruction_with_aliases(
                turn_type, target_road_aliases, int_aliases, from_road_aliases
            ),
            
            # USER-FRIENDLY NAVIGATION PHRASES
            'user_navigation_phrases': {
                'simple': f'{self.turn_type_to_phrase(turn_type)} to {target_road_aliases.get("primary_name", f"Road {to_road_id}")}',
                'with_landmark': f'{self.turn_type_to_phrase(turn_type)} at {int_aliases.get("primary_name", "intersection")} to reach {target_road_aliases.get("primary_name", f"Road {to_road_id}")}',
                'detailed': f'from {from_road_aliases.get("primary_name", f"Road {from_road_id}")}, {self.turn_type_to_phrase(turn_type)} at {int_aliases.get("primary_name", "the intersection")} to get onto {target_road_aliases.get("primary_name", f"Road {to_road_id}")}',
                'casual': f'{self.turn_type_to_phrase(turn_type)} to get on {target_road_aliases.get("user_friendly_references", {}).get("casual", target_road_aliases.get("primary_name", f"Road {to_road_id}"))}'
            },
            
            # Clock directions for precise navigation
            'from_clock': self.get_clock_direction(int_center, from_road['points'][-1]),
            'to_clock': self.get_clock_direction(int_center, target_road['points'][0]),
            
            # NARRATIVE RELATIONSHIP
            'branch_narrative': {
                'connection_type': f'{turn_type} turn',
                'from_road': from_road_aliases.get('primary_name', f'Road {from_road_id}'),
                'to_road': target_road_aliases.get('primary_name', f'Road {to_road_id}'),
                'via_intersection': int_aliases.get('primary_name', f'Intersection {intersection["id"]}'),
                'user_story': f'this branch allows you to go from {from_road_aliases.get("primary_name", f"Road {from_road_id}")} to {target_road_aliases.get("primary_name", f"Road {to_road_id}")} by {self.turn_type_to_phrase(turn_type)} at {int_aliases.get("primary_name", "the intersection")}',
                'landmark_references': list(set(int_aliases.get('visual_landmarks', []) + target_road_aliases.get('visual_landmarks', [])))
            }
        }
        
        return branch

    def generate_navigation_instruction_with_aliases(self, turn_type, target_road_aliases, int_aliases, from_road_aliases):
        """Generate comprehensive navigation instruction with user-friendly aliases"""
        turn_instruction = self.turn_type_to_phrase(turn_type).capitalize()
        
        # Add intersection name
        intersection_name = int_aliases.get('primary_name', 'intersection')
        turn_instruction += f" at {intersection_name}"
        
        # Add target road name  
        road_name = target_road_aliases.get('primary_name', '')
        if road_name:
            turn_instruction += f" onto {road_name}"
        
        # Add edge information with user context
        edge_context = target_road_aliases.get('edge_context', {})
        if edge_context.get('is_entry_exit', False):
            turn_instruction += f" ({edge_context.get('user_navigation', 'leads to edge')})"
        
        # Add landmark reference
        landmarks = int_aliases.get('visual_landmarks', [])
        if landmarks:
            turn_instruction += f" near {landmarks[0]}"
        
        return turn_instruction

    def calculate_turn_type(self, from_road, to_road, intersection_center):
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

    def get_clock_direction(self, center, point):
        """Get clock direction from center to point"""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = atan2(-dy, dx)
        
        angle_deg = degrees(angle)
        if angle_deg < 0:
            angle_deg += 360
        
        clock_hour = int((angle_deg + 15) / 30) % 12
        return clock_hour

    def get_intersection_metadata_for_lane(self, road_id, lane_points):
        """Get intersection metadata for lane ending"""
        if not lane_points:
            return None
        
        lane_end = lane_points[-1]
        
        # Find closest intersection
        closest_intersection = None
        min_distance = float('inf')
        
        for intersection in self.intersections:
            center = intersection['center']
            dist = sqrt((lane_end[0] - center[0])**2 + (lane_end[1] - center[1])**2)
            
            if dist <= INTERSECTION_RADIUS and dist < min_distance:
                min_distance = dist
                closest_intersection = intersection
        
        if closest_intersection:
            return closest_intersection.get('metadata', {})
        
        return None

    def save_integrated_outputs(self):
        """Save all outputs with consistent IDs, comprehensive metadata, and user-friendly aliases"""
        print("\n💾 SAVING INTEGRATED OUTPUTS WITH USER-FRIENDLY ALIASES...")
        print("-" * 60)
        
        # Collect comprehensive statistics
        total_branches = sum(len(tree['branches']) for tree in self.lane_trees)
        trees_with_branches = sum(1 for tree in self.lane_trees if tree['branches'])
        edge_trees = sum(1 for tree in self.lane_trees 
                        if tree['metadata']['edge_analysis']['has_edge_connection'])
        
        all_landmarks = set()
        all_businesses = set()
        all_edge_ids = set()
        
        for tree in self.lane_trees:
            all_landmarks.update(tree['metadata'].get('landmarks', []))
            for branch in tree['branches']:
                all_landmarks.update(branch.get('intersection_landmarks', []))
                all_businesses.update(branch.get('intersection_businesses', []))
            
            # Collect edge IDs
            edge_analysis = tree['metadata']['edge_analysis']
            start_edge = edge_analysis.get('start_edge', {})
            end_edge = edge_analysis.get('end_edge', {}) 
            if start_edge.get('edge_id'):
                all_edge_ids.add(start_edge['edge_id'])
            if end_edge.get('edge_id'):
                all_edge_ids.add(end_edge['edge_id'])

        # Calculate edge analysis summary with user-friendly names
        self.edge_analysis_summary = {
            "total_roads": len(self.roads),
            "roads_with_edge_connections": len(self.geographic_road_map['west'] + 
                                               self.geographic_road_map['east'] + 
                                               self.geographic_road_map['north'] + 
                                               self.geographic_road_map['south']),
            "total_intersections": len(self.intersections),
            "intersections_at_edges": sum(1 for i in self.intersections 
                                         if i['metadata']['edge_analysis']['is_edge_intersection']),
            "all_edge_ids": sorted(list(all_edge_ids)),
            "edge_distribution": {
                direction: [{'edge_id': f"{direction}_end_{i:02d}",
                           'road_name': self.common_language_vocabulary['roads'].get(rid, {}).get('primary_name', f'Road {rid}'),
                           'user_friendly_name': self.common_language_vocabulary['roads'].get(rid, {}).get('primary_name', f'Road {rid}'),
                           'point_type': 'start' if direction in next(r for r in self.roads if r['id'] == rid)['metadata']['edge_analysis']['true_start_directions'] else 'end'}
                          for i, rid in enumerate(road_ids)]
                for direction, road_ids in self.geographic_road_map.items() if road_ids
            },
            "canvas_size": CANVAS_SIZE,
            "edge_tolerance": EDGE_TOLERANCE
        }

        # Create comprehensive integrated data with COMMON LANGUAGE VOCABULARY
        integrated_data = {
            "roads": self.roads,
            "intersections": self.intersections,
            "bedrock_analysis": self.bedrock_metadata,
            
            # COMMON LANGUAGE VOCABULARY - Core addition from PDF requirements
            "common_language_vocabulary": self.common_language_vocabulary,
            
            # Navigation graph with consistent IDs
            "navigation_graph": {
                "road_to_intersections": self.road_to_intersections,
                "intersection_to_roads": self.intersection_to_roads,
                "lane_to_road": self.lane_to_road
            },
            
            # Edge analysis
            "edge_analysis_summary": self.edge_analysis_summary,
            
            # USER-FRIENDLY QUICK REFERENCE - Easy lookup for AI systems
            "user_friendly_quick_reference": {
                "road_names": {str(road_id): aliases['primary_name'] for road_id, aliases in self.common_language_vocabulary['roads'].items()},
                "intersection_names": {str(int_id): aliases['primary_name'] for int_id, aliases in self.common_language_vocabulary['intersections'].items()},
                "lane_names": {lane_id: aliases['primary_name'] for lane_id, aliases in self.common_language_vocabulary['lanes'].items()},
                "all_landmarks": list(self.common_language_vocabulary['landmarks'].keys()),
                "spatial_relationships_count": len(self.common_language_vocabulary['spatial_relationships'])
            },
            
            # Comprehensive metadata
            "metadata": {
                "total_roads": len(self.roads),
                "total_intersections": len(self.intersections),
                "coordinate_system": "image_pixels",
                "origin": "top_left",
                "processing_parameters": {
                    "min_line_length": MIN_LINE_LENGTH,
                    "canvas_size": CANVAS_SIZE,
                    "edge_tolerance": EDGE_TOLERANCE,
                    "intersection_radius": INTERSECTION_RADIUS,
                    "lane_offset_px": LANE_OFFSET_PX
                },
                "road_classes": list(set([road["metadata"]["road_class"] for road in self.roads])),
                "intersection_types": list(set([intersection["metadata"]["intersection_type"] for intersection in self.intersections])),
                "navigation_features": {
                    "supports_turn_by_turn": True,
                    "supports_landmark_navigation": True,
                    "supports_route_planning": True,
                    "supports_narrative_parsing": True,
                    "supports_edge_aware_navigation": True,
                    "consistent_id_system": True,
                    "supports_user_friendly_aliases": True,
                    "supports_common_language_vocabulary": True,
                    "supports_spatial_relationship_queries": True
                }
            }
        }

        # Lane tree data with user-friendly aliases
        lane_tree_data = {
            "lane_trees": self.lane_trees,
            "road_connections": {str(k): v for k, v in self.intersection_to_roads.items()},
            "bedrock_analysis": self.bedrock_metadata,
            
            # COMMON LANGUAGE VOCABULARY
            "common_language_vocabulary": self.common_language_vocabulary,
            
            # Enhanced navigation metadata with user-friendly names
            "navigation_metadata": {
                "all_landmarks": list(all_landmarks),
                "all_businesses": list(all_businesses),
                "road_names": [self.common_language_vocabulary['roads'].get(road['id'], {}).get('primary_name', f'Road {road["id"]}') for road in self.roads],
                "intersection_types": list(set(i['metadata']['intersection_type'] for i in self.intersections)),
                "edge_entry_points": self.edge_entry_points,
                "roads_with_edges": {str(rid): {
                    'road_name': self.common_language_vocabulary['roads'].get(rid, {}).get('primary_name', f'Road {rid}'),
                    'edge_sides': next(r for r in self.roads if r['id'] == rid)['metadata']['edge_analysis']['edge_sides']
                } for direction_roads in self.geographic_road_map.values() for rid in direction_roads},
                "all_edge_ids": sorted(list(all_edge_ids)),
                "user_friendly_names_count": {
                    "roads": len(self.common_language_vocabulary['roads']),
                    "intersections": len(self.common_language_vocabulary['intersections']),
                    "lanes": len(self.common_language_vocabulary['lanes']),
                    "landmarks": len(self.common_language_vocabulary['landmarks'])
                }
            },
            
            # Statistics with user-friendly context
            "statistics": {
                "total_trees": len(self.lane_trees),
                "total_branches": total_branches,
                "trees_with_branches": trees_with_branches,
                "average_branches_per_tree": total_branches / len(self.lane_trees) if self.lane_trees else 0,
                "roads_count": len(self.roads),
                "intersections_count": len(self.intersections),
                "unique_landmarks": len(all_landmarks),
                "unique_businesses": len(all_businesses),
                "edge_trees": edge_trees,
                "entry_point_trees": sum(1 for tree in self.lane_trees 
                                       if tree['metadata']['edge_analysis']['is_geographic_entry_point']),
                "exit_point_trees": sum(1 for tree in self.lane_trees 
                                      if tree['metadata']['edge_analysis']['is_geographic_exit_point']),
                "unique_edge_ids": len(all_edge_ids),
                "spatial_relationships": len(self.common_language_vocabulary['spatial_relationships']),
                "user_friendly_aliases_generated": {
                    "total_road_aliases": len(self.common_language_vocabulary['roads']),
                    "total_intersection_aliases": len(self.common_language_vocabulary['intersections']),
                    "total_lane_aliases": len(self.common_language_vocabulary['lanes']),
                    "total_landmarks": len(self.common_language_vocabulary['landmarks'])
                }
            },
            
            "edge_analysis_summary": self.edge_analysis_summary,
            "parameters": {
                "lane_offset_px": LANE_OFFSET_PX,
                "intersection_radius": INTERSECTION_RADIUS
            },
            "metadata": {
                "coordinate_system": "image_pixels",
                "origin": "top_left",
                "supports_narrative_parsing": True,
                "supports_landmark_navigation": True,
                "supports_turn_by_turn": True,
                "supports_edge_aware_navigation": True,
                "consistent_id_system": True,
                "supports_user_friendly_aliases": True,
                "supports_common_language_vocabulary": True,
                "ai_human_common_language_ready": True,
                "step_0_compliance": "Implements user-friendly aliases and meaning co-creation as per PDF Step 0"
            }
        }

        # Save all files locally
        with open(OUTPUT_INTEGRATED_JSON, 'w') as f:
            json.dump(integrated_data, f, indent=2)
        
        with open(OUTPUT_CENTERLINES_JSON, 'w') as f:
            json.dump(integrated_data, f, indent=2)
        
        with open(OUTPUT_INTERSECTIONS_JSON, 'w') as f:
            json.dump({"intersections": self.intersections}, f, indent=2)
        
        with open(OUTPUT_LANE_TREES_JSON, 'w') as f:
            json.dump(lane_tree_data, f, indent=2)

        print(f"✅ Saved integrated network to {OUTPUT_INTEGRATED_JSON}")
        print(f"✅ Saved centerlines to {OUTPUT_CENTERLINES_JSON}")
        print(f"✅ Saved intersections to {OUTPUT_INTERSECTIONS_JSON}")
        print(f"✅ Saved lane trees to {OUTPUT_LANE_TREES_JSON}")

        # Upload all outputs to S3
        print("\n🌐 UPLOADING TO S3...")
        print("-" * 40)
        
        # Define S3 keys with connection_id folder structure
        s3_prefix = f"outputs/{self.connection_id}/"
        
        # Upload JSON files
        self.upload_json_to_s3(integrated_data, f"{s3_prefix}{OUTPUT_INTEGRATED_JSON}")
        self.upload_json_to_s3(integrated_data, f"{s3_prefix}{OUTPUT_CENTERLINES_JSON}")
        self.upload_json_to_s3({"intersections": self.intersections}, f"{s3_prefix}{OUTPUT_INTERSECTIONS_JSON}")
        self.upload_json_to_s3(lane_tree_data, f"{s3_prefix}{OUTPUT_LANE_TREES_JSON}")
        
        # Upload image files
        if os.path.exists(OUTPUT_IMG):
            self.upload_to_s3(OUTPUT_IMG, f"{s3_prefix}{OUTPUT_IMG}")
        
        if os.path.exists(DEBUG_SKELETON):
            self.upload_to_s3(DEBUG_SKELETON, f"{s3_prefix}{DEBUG_SKELETON}")
        
        print(f"✅ All outputs uploaded to S3 under folder: {s3_prefix}")

    def visualize_integrated_network(self):
        """Create comprehensive visualization of integrated network with user-friendly labels"""
        canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)

        # Colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Draw roads with edge highlighting and user-friendly names
        for i, road in enumerate(self.roads):
            color = colors[i % len(colors)]
            points = road["points"]
            
            # Highlight edge-connected roads
            edge_analysis = road['metadata']['edge_analysis']
            line_thickness = 6 if edge_analysis['has_edge_connection'] else 3
            
            for j in range(len(points) - 1):
                pt1 = (int(points[j][0]), int(points[j][1]))
                pt2 = (int(points[j+1][0]), int(points[j+1][1]))
                cv2.line(canvas, pt1, pt2, color, line_thickness)
            
            # Mark edge points
            if points:
                start_pt = (int(points[0][0]), int(points[0][1]))
                end_pt = (int(points[-1][0]), int(points[-1][1]))
                
                start_edge = edge_analysis.get('start_edge', {})
                end_edge = edge_analysis.get('end_edge', {})
                
                if start_edge.get('is_edge', False):
                    cv2.circle(canvas, start_pt, 12, (0, 255, 0), -1)
                    if start_edge.get('edge_id'):
                        cv2.putText(canvas, start_edge['edge_id'], 
                                   (start_pt[0] + 15, start_pt[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    cv2.circle(canvas, start_pt, 6, (255, 255, 255), -1)
                
                if end_edge.get('is_edge', False):
                    cv2.circle(canvas, end_pt, 12, (0, 0, 255), -1)
                    if end_edge.get('edge_id'):
                        cv2.putText(canvas, end_edge['edge_id'], 
                                   (end_pt[0] + 15, end_pt[1] + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.circle(canvas, end_pt, 6, (0, 0, 0), -1)
                
                # Road label with USER-FRIENDLY NAME
                mid_idx = len(points) // 2
                mid_pt = (int(points[mid_idx][0]), int(points[mid_idx][1]))
                
                # Use user-friendly name instead of generic road name
                road_aliases = self.common_language_vocabulary['roads'].get(road['id'], {})
                user_friendly_name = road_aliases.get('primary_name', f'Road {road["id"]}')
                road_id = road['id']
                
                edge_indicator = ""
                if edge_analysis['has_edge_connection']:
                    edge_sides = '/'.join(edge_analysis['edge_sides'][:2])
                    edge_indicator = f"[{edge_sides}]"
                
                label = f"[{road_id}]{user_friendly_name}{edge_indicator}"
                cv2.putText(canvas, label, (mid_pt[0], mid_pt[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw intersections with user-friendly names
        for intersection in self.intersections:
            center_x, center_y = intersection['center']
            int_id = intersection['id']
            metadata = intersection['metadata']
            
            radius = 8 + metadata['connected_roads_count']
            cv2.circle(canvas, (int(center_x), int(center_y)), radius, (0, 255, 255), -1)
            cv2.circle(canvas, (int(center_x), int(center_y)), radius, (255, 255, 255), 2)
            
            # Intersection label with USER-FRIENDLY NAME
            int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
            user_friendly_name = int_aliases.get('primary_name', f'Intersection {int_id}')
            int_type = metadata['intersection_type']
            
            label = f"[{int_id}]{user_friendly_name}"
            
            # Add edge indicator
            edge_analysis = metadata.get('edge_analysis', {})
            if edge_analysis.get('is_edge_intersection', False):
                edge_sides = '/'.join(edge_analysis.get('edge_sides', []))
                label += f"[{edge_sides}]"
            
            cv2.putText(canvas, label,
                       (int(center_x) + radius + 5, int(center_y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Title with statistics including user-friendly info
        title = f"Integrated Network with User-Friendly Aliases: {len(self.roads)} roads, {len(self.intersections)} intersections"
        cv2.putText(canvas, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        edge_stats = f"Edge connections: {self.edge_analysis_summary['roads_with_edge_connections']} roads, {len(self.edge_analysis_summary['all_edge_ids'])} IDs"
        cv2.putText(canvas, edge_stats, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        alias_info = f"User-friendly aliases: {len(self.common_language_vocabulary['roads'])} roads, {len(self.common_language_vocabulary['intersections'])} intersections"
        cv2.putText(canvas, alias_info, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        consistency_info = "✅ Common Language Ready for AI-Human Communication"
        cv2.putText(canvas, consistency_info, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(OUTPUT_IMG, canvas)
        print(f"✅ Saved integrated network visualization with user-friendly labels to {OUTPUT_IMG}")

    def process_complete_integrated_network(self):
        """Complete integrated processing pipeline with user-friendly aliases"""
        print("🚗 INTEGRATED ROAD NETWORK GENERATOR WITH COMMON LANGUAGE")
        print("=" * 80)
        print("Creating user-friendly aliases and narrative relationships for AI-Human communication")
        print("Implementing Step 0 from PDF: Map Structure Analysis and Meaning Co-creation")
        print(f"📁 Connection ID: {self.connection_id}")
        print(f"🪣 S3 Bucket: {self.bucket_name}")
        
        # Step 1: Extract skeleton and basic network
        print("\n📍 STEP 1: Extracting road network skeleton...")
        skeleton = self.extract_medial_axis(MASK_PATH)
        
        # Step 2: Trace roads and find intersections with consistent IDs
        print("\n🛣️  STEP 2: Tracing roads and intersections with consistent IDs...")
        self.roads, endpoints, junctions = self.trace_skeleton_paths(skeleton)
        self.intersections = self.find_major_intersections(junctions, skeleton)
        
        # Step 3: Analyze images with Bedrock for user-friendly context
        print("\n🧠 STEP 3: Analyzing images with AI for user-friendly context...")
        self.bedrock_metadata = self.analyze_roadmap_with_bedrock(ROADMAP_PATH, SATELLITE_PATH)
        
        # Step 4: Build consistent cross-references
        print("\n🔗 STEP 4: Building consistent cross-references...")
        self.build_consistent_road_intersection_mapping()
        
        # Step 5: GENERATE COMMON LANGUAGE VOCABULARY - Core PDF requirement
        print("\n🗣️  STEP 5: Generating common language vocabulary (PDF Step 0)...")
        self.generate_common_language_vocabulary()
        
        # Step 6: Enhance with integrated metadata including user-friendly aliases
        print("\n🎯 STEP 6: Enhancing with integrated metadata and user-friendly aliases...")
        self.enhance_with_integrated_metadata()
        
        # Step 7: Generate lane trees with user-friendly aliases
        print("\n🛤️  STEP 7: Generating lane trees with user-friendly aliases...")
        self.generate_integrated_lane_trees()
        
        # Step 8: Save all outputs with common language vocabulary
        print("\n💾 STEP 8: Saving integrated outputs with common language vocabulary...")
        self.save_integrated_outputs()
        
        # Step 9: Create visualization with user-friendly names
        print("\n🎨 STEP 9: Creating visualization with user-friendly names...")
        self.visualize_integrated_network()
        
        # Final summary
        print(f"\n🎉 INTEGRATED NETWORK WITH COMMON LANGUAGE COMPLETE!")
        print("=" * 80)
        print(f"✅ Roads: {len(self.roads)} (with user-friendly aliases)")
        print(f"✅ Intersections: {len(self.intersections)} (with user-friendly aliases)")
        print(f"✅ Lane trees: {len(self.lane_trees)} (with user-friendly aliases)")
        print(f"✅ Edge connections: {self.edge_analysis_summary['roads_with_edge_connections']} roads")
        print(f"✅ Common language vocabulary generated:")
        print(f"    - Road aliases: {len(self.common_language_vocabulary['roads'])}")
        print(f"    - Intersection aliases: {len(self.common_language_vocabulary['intersections'])}")
        print(f"    - Lane aliases: {len(self.common_language_vocabulary['lanes'])}")
        print(f"    - Spatial relationships: {len(self.common_language_vocabulary['spatial_relationships'])}")
        print(f"    - Landmark details: {len(self.common_language_vocabulary['landmarks'])}")
        print(f"✅ Cross-references: All IDs are consistent across all JSON files")
        print(f"✅ AI Analysis: {len(self.bedrock_metadata.get('user_friendly_roads', []))} user-friendly road descriptions")
        print(f"🌐 All outputs uploaded to S3: s3://{self.bucket_name}/outputs/{self.connection_id}/")
        print(f"\n🎯 READY FOR NARRATIVE-BASED ROUTE GENERATION!")
        print(f"   ✨ AI and humans can now use the same language to describe:")
        print(f"      - Roads: 'the main street', 'narrow road behind station'")
        print(f"      - Intersections: 'intersection by the station', 'big crossing'")
        print(f"      - Lanes: 'right lane heading to station', 'westbound main road'")
        print(f"      - Navigation: 'turn right at the station intersection onto the main road'")
        print(f"   📋 Implements PDF Step 0: Structure Analysis and Meaning Co-creation")
        print(f"   🤝 Common vocabulary established between AI and human users")


def main():
    connection_id = sys.argv[1] if len(sys.argv) > 1 else None
    generator = IntegratedRoadNetworkGenerator(connection_id)
    generator.process_complete_integrated_network()

if __name__ == '__main__':
    main()
```

## Key Additions Based on the PDF:

### 1. **User-Friendly Aliases** (Step 0 from PDF)
- Added `common_language_vocabulary` dictionary with aliases for roads, intersections, and lanes
- Each entity now has multiple user-friendly names like "main street", "the wide road", "intersection by the station"
- Generated natural descriptions that users would actually say

### 2. **Narrative Relationships** (Semantic Understanding)
- Added spatial relationships between roads, intersections, and lanes
- Implemented connection narratives like "Road 1 connects to Road 2 via Intersection A" 
- Added landmark-based relationships: "lane in front of the shop", "road that passes the supermarket"

### 3. **Enhanced Bedrock Analysis**
- Modified prompt to specifically ask for user-friendly descriptions and spatial relationships
- Focus on how humans would naturally describe roads and intersections
- Extract landmark-based navigation references

### 4. **Semantic Metadata Structure** 
- `user_friendly_aliases` in every road/intersection/lane object
- `narrative_relationships` describing connections in natural language
- `common_language` section with different levels of formality (formal, casual, descriptive)

### 5. **Step 0 Implementation** 
- Implements "Map Structure Analysis and Meaning Co-creation" from the PDF
- Creates shared vocabulary between AI and humans
- Enables natural language route descriptions like "turn right at the station intersection onto the main road"

This creates the foundational "common language" that allows AI systems to understand human navigation descriptions and provide human-friendly responses, exactly as described in Step 0 of the PDF document.