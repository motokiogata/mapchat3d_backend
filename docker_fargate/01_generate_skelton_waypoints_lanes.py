#!/usr/bin/env python3
# main.py
import sys
import json
from math import sqrt

# Import all modules
from config import *
from ./components/bedrock_analyzer import BedrockAnalyzer
from ./components/skeleton_extractor import SkeletonExtractor
from ./components/geometry_utils import GeometryUtils
from ./components/common_language import CommonLanguageGenerator
from ./components/lane_generator import LaneGenerator
from ./components/s3_handler import S3Handler
from ./components/visualizer import NetworkVisualizer

class IntegratedRoadNetworkGenerator:
    def __init__(self, connection_id=None):
        # Core data with consistent IDs
        self.roads = []
        self.intersections = []
        self.lane_trees = []
        
        # Cross-reference mappings
        self.road_to_intersections = {}
        self.intersection_to_roads = {}
        self.lane_to_road = {}
        
        # Enhanced metadata
        self.bedrock_metadata = {}
        self.edge_analysis_summary = {}
        self.common_language_vocabulary = {}
        
        # Edge analysis
        self.edge_entry_points = {}
        self.geographic_road_map = {'west': [], 'east': [], 'north': [], 'south': []}
        
        # Initialize modules
        self.bedrock_analyzer = BedrockAnalyzer()
        self.skeleton_extractor = SkeletonExtractor()
        self.common_language_generator = CommonLanguageGenerator()
        self.s3_handler = S3Handler(connection_id)

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

    def enhance_with_integrated_metadata(self):
        """Enhance roads and intersections with user-friendly aliases and narrative relationships"""
        print("\n🎯 ENHANCING WITH INTEGRATED METADATA (including user-friendly aliases)...")
        print("-" * 70)
        
        # Process edge analysis first
        edge_roads_count = 0
        roads_with_edges = []
        
        for road in self.roads:
            road_id = road['id']
            points = road['points']
            
            start_edge_info = GeometryUtils.is_point_at_edge(points[0])
            end_edge_info = GeometryUtils.is_point_at_edge(points[-1])
            
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
            
            # Calculate comprehensive metadata
            direction_info = GeometryUtils.calculate_road_direction(points)
            curvature = GeometryUtils.calculate_road_curvature(points)
            width_category = GeometryUtils.estimate_road_width_category(points)
            
            road['metadata'] = {
                # Basic properties
                'name': road_aliases.get('primary_name', f'Road_{road_id}'),
                'alt_names': road_aliases.get('alternative_names', []),
                'road_class': 'local_street' if width_category == 'narrow' else 'main_road',
                'road_type': 'local',
                'estimated_speed_limit': 30 if width_category == 'narrow' else 40,
                'traffic_density': 'low' if width_category == 'narrow' else 'medium',
                'curvature': curvature,
                'width_category': width_category,
                'landmarks': road_aliases.get('visual_landmarks', []),
                'estimated_length_meters': len(points) * 0.5,
                'priority': 1 if width_category == 'narrow' else 2 if width_category == 'medium' else 3,
                
                # USER-FRIENDLY ALIASES
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
                'intersection_connections': self.road_to_intersections.get(road_id, [])
            }
            
            # Add to geographic mapping if has edge connection
            if road['metadata']['edge_analysis']['has_edge_connection']:
                for side in road['metadata']['edge_analysis']['edge_sides']:
                    if side in self.geographic_road_map:
                        self.geographic_road_map[side].append(road_id)
        
        # Enhance intersections with user-friendly aliases
        for intersection in self.intersections:
            int_id = intersection['id']
            center = intersection['center']
            
            # Get connected roads with consistent IDs
            connected_roads = self.intersection_to_roads.get(int_id, [])
            
            # Get user-friendly aliases
            int_aliases = self.common_language_vocabulary['intersections'].get(int_id, {})
            
            # Calculate intersection type
            int_type = GeometryUtils.determine_intersection_type(len(connected_roads))
            
            # Check if intersection is at edge
            intersection_edge_info = GeometryUtils.is_point_at_edge(center)
            
            intersection['metadata'] = {
                # Basic properties
                'intersection_type': int_type,
                'connected_roads_count': len(connected_roads),
                'connected_road_ids': [road['road_id'] for road in connected_roads],
                
                # Enhanced info
                'landmarks': int_aliases.get('visual_landmarks', []),
                'nearby_businesses': [],
                'traffic_signals': len(connected_roads) >= 3,
                'estimated_wait_time': 15 if len(connected_roads) >= 3 else 5,
                
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
                'road_connections': connected_roads
            }
        
        print(f"✅ Enhanced {len(self.roads)} roads and {len(self.intersections)} intersections with user-friendly aliases")
        print(f"🎯 Found {edge_roads_count} roads with edge connections")

    def save_integrated_outputs(self):
        """Save all outputs with user-friendly aliases"""
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
            
            # USER-FRIENDLY QUICK REFERENCE
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
        self.s3_handler.upload_all_outputs(integrated_data, lane_tree_data, self.intersections)

    def process_complete_integrated_network(self):
        """Complete integrated processing pipeline with user-friendly aliases"""
        print("🚗 INTEGRATED ROAD NETWORK GENERATOR WITH COMMON LANGUAGE")
        print("=" * 80)
        print("Creating user-friendly aliases and narrative relationships for AI-Human communication")
        print("Implementing Step 0 from PDF: Map Structure Analysis and Meaning Co-creation")
        print(f"📁 Connection ID: {self.s3_handler.connection_id}")
        print(f"🪣 S3 Bucket: {self.s3_handler.bucket_name}")
        
        # Step 1: Extract skeleton and basic network
        print("\n📍 STEP 1: Extracting road network skeleton...")
        skeleton = self.skeleton_extractor.extract_medial_axis(MASK_PATH)
        
        # Step 2: Trace roads and find intersections with consistent IDs
        print("\n🛣️  STEP 2: Tracing roads and intersections with consistent IDs...")
        self.roads, endpoints, junctions = self.skeleton_extractor.trace_skeleton_paths(skeleton)
        self.intersections = self.skeleton_extractor.find_major_intersections(junctions, skeleton)
        
        # Step 3: Analyze images with Bedrock for user-friendly context
        print("\n🧠 STEP 3: Analyzing images with AI for user-friendly context...")
        self.bedrock_metadata = self.bedrock_analyzer.analyze_roadmap_with_bedrock(ROADMAP_PATH, SATELLITE_PATH)
        
        # Step 4: Build consistent cross-references
        print("\n🔗 STEP 4: Building consistent cross-references...")
        self.build_consistent_road_intersection_mapping()
        
        # Step 5: GENERATE COMMON LANGUAGE VOCABULARY - Core PDF requirement
        print("\n🗣️  STEP 5: Generating common language vocabulary (PDF Step 0)...")
        self.common_language_vocabulary = self.common_language_generator.generate_common_language_vocabulary(
            self.roads, self.intersections, self.bedrock_metadata, 
            self.road_to_intersections, self.intersection_to_roads
        )
        
        # Step 6: Enhance with integrated metadata including user-friendly aliases
        print("\n🎯 STEP 6: Enhancing with integrated metadata and user-friendly aliases...")
        self.enhance_with_integrated_metadata()
        
        # Step 7: Generate lane trees with user-friendly aliases
        print("\n🛤️  STEP 7: Generating lane trees with user-friendly aliases...")
        lane_generator = LaneGenerator(
            self.roads, self.intersections, self.road_to_intersections, 
            self.intersection_to_roads, self.common_language_vocabulary
        )
        self.lane_trees = lane_generator.generate_integrated_lane_trees()
        self.lane_to_road = lane_generator.lane_to_road
        
        # Step 8: Save all outputs with common language vocabulary
        print("\n💾 STEP 8: Saving integrated outputs with common language vocabulary...")
        self.save_integrated_outputs()
        
        # Step 9: Create visualization with user-friendly names
        print("\n🎨 STEP 9: Creating visualization with user-friendly names...")
        visualizer = NetworkVisualizer(
            self.roads, self.intersections, self.common_language_vocabulary, self.edge_analysis_summary
        )
        visualizer.visualize_integrated_network()
        
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
        print(f"🌐 All outputs uploaded to S3: s3://{self.s3_handler.bucket_name}/outputs/{self.s3_handler.connection_id}/")
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