# common_language.py
from geometry_utils import GeometryUtils

class CommonLanguageGenerator:
    def __init__(self):
        self.common_language_vocabulary = {
            'roads': {},
            'intersections': {}, 
            'lanes': {},
            'landmarks': {},
            'spatial_relationships': {}
        }

    def generate_common_language_vocabulary(self, roads, intersections, bedrock_metadata, 
                                          road_to_intersections, intersection_to_roads):
        """Generate user-friendly aliases and spatial relationships based on Bedrock analysis"""
        print("\n🗣️  GENERATING COMMON LANGUAGE VOCABULARY...")
        print("-" * 60)
        
        bedrock_roads = bedrock_metadata.get('user_friendly_roads', [])
        bedrock_intersections = bedrock_metadata.get('user_friendly_intersections', [])
        bedrock_lanes = bedrock_metadata.get('user_friendly_lanes', [])
        bedrock_relationships = bedrock_metadata.get('spatial_relationships', [])
        bedrock_landmarks = bedrock_metadata.get('landmark_details', [])
        
        # Generate road aliases
        for road in roads:
            road_id = road['id']
            
            # Try to match with Bedrock analysis
            bedrock_road = None
            if road_id < len(bedrock_roads):
                bedrock_road = bedrock_roads[road_id]
            elif bedrock_roads:
                bedrock_road = bedrock_roads[0]
            
            # Calculate road characteristics for fallback descriptions
            direction_info = GeometryUtils.calculate_road_direction(road['points'])
            curvature = GeometryUtils.calculate_road_curvature(road['points'])
            width_category = GeometryUtils.estimate_road_width_category(road['points'])
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
        for intersection in intersections:
            int_id = intersection['id']
            
            # Try to match with Bedrock analysis
            bedrock_intersection = None
            if int_id < len(bedrock_intersections):
                bedrock_intersection = bedrock_intersections[int_id]
            elif bedrock_intersections:
                bedrock_intersection = bedrock_intersections[0]
            
            # Get connected roads for context
            connected_roads = intersection_to_roads.get(int_id, [])
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
        for road in roads:
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
                        'connects_to': [conn['intersection_id'] for conn in road_to_intersections.get(road_id, [])]
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
        for road_id, intersections_list in road_to_intersections.items():
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
        for int_id, roads_list in intersection_to_roads.items():
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
                    road_1 = next(r for r in roads if r['id'] == road_1_id)
                    road_2 = next(r for r in roads if r['id'] == road_2_id)
                    intersection = next(i for i in intersections if i['id'] == int_id)
                    turn_type = GeometryUtils.calculate_turn_type(road_1, road_2, intersection['center'])
                    
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
        
        return self.common_language_vocabulary

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