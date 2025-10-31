#analytics.py
import json
import boto3
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))
ecs_client = boto3.client('ecs', region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))  # ADD THIS LINE
table = dynamodb.Table("AccidentDataTable")

class InteractiveTrafficAnalyzer:

    def _identify_missing_fields(self, route_json: dict) -> list:
        """
        Identify which fields are missing or unknown in the route JSON
        """
        missing = []
        
        def check_dict(d, path=""):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key
                
                if value is None or value == "unknown" or value == -1:
                    missing.append(current_path)
                elif isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            check_dict(item, f"{current_path}[{i}]")
        
        check_dict(route_json)
        return missing


    def enhance_missing_values_with_llm(self, route_json):
        """
        SIMPLIFIED: Only enhance critical missing values, not everything
        """
        
        # Check ONLY critical fields
        critical_fields = {
            "user_origin_direction": route_json.get("vehicles", {}).get("user_vehicle", {}).get("path", {}).get("origin", {}).get("direction"),
            "other_origin_direction": route_json.get("vehicles", {}).get("other_vehicle", {}).get("path", {}).get("origin", {}).get("direction"),
            "collision_type": route_json.get("collision", {}).get("type")
        }
        
        missing_critical = [k for k, v in critical_fields.items() if v == "unknown" or v is None]
        
        if not missing_critical:
            logger.info("✅ All critical fields present")
            return route_json
        
        logger.info(f"🔍 Enhancing {len(missing_critical)} critical fields: {missing_critical}")
        
        # Prepare MINIMAL context
        context = self._prepare_comprehensive_context()
        
        prompt = f"""
        Fill ONLY these specific missing values in this accident analysis:
        
        CONVERSATION CONTEXT:
        {context}
        
        MISSING VALUES TO FILL:
        {json.dumps(missing_critical, indent=2)}
        
        AVAILABLE DATA:
        - User road: {route_json.get("vehicles", {}).get("user_vehicle", {}).get("path", {}).get("origin", {}).get("road_name")}
        - Other vehicle road: {route_json.get("vehicles", {}).get("other_vehicle", {}).get("path", {}).get("origin", {}).get("road_name")}
        
        OUTPUT FORMAT (JSON only):
        {{
            "user_origin_direction": "southeast",
            "other_origin_direction": "northwest",
            "collision_type": "intersection_collision"
        }}
        
        No other text. Just the JSON object with these 3 fields.
        """
        
        try:
            response = self.call_bedrock(prompt, max_tokens=200)  # Much smaller!
            result = self._extract_json_from_response(response)
            
            if result:
                # Apply the enhancements
                if "user_origin_direction" in result:
                    route_json["vehicles"]["user_vehicle"]["path"]["origin"]["direction"] = result["user_origin_direction"]
                
                if "other_origin_direction" in result:
                    route_json["vehicles"]["other_vehicle"]["path"]["origin"]["direction"] = result["other_origin_direction"]
                
                if "collision_type" in result:
                    route_json["collision"]["type"] = result["collision_type"]
                
                logger.info("✅ LLM enhanced critical fields")
            
            return route_json
                
        except Exception as e:
            logger.error(f"❗ LLM enhancement failed: {e}")
            return route_json  # Return original if fails


    def _prepare_comprehensive_context(self):
        """Prepare all available context for LLM"""
        context_parts = []
        
        # User responses
        context_parts.append("USER RESPONSES:")
        for key, value in self.responses.items():
            context_parts.append(f"  {key}: {value}")
        
        # Key relationships mentioned
        context_parts.append(f"\nUSER PATH DATA:")
        context_parts.append(f"  Origin direction input: {self.user_path.get('origin_direction')}")
        context_parts.append(f"  Intended maneuver: {self.user_path.get('intended_maneuver')}")
        context_parts.append(f"  Other vehicle position: {(self.user_path.get('other_vehicle', {}) or {}).get('position')}")
        context_parts.append(f"  Other vehicle action: {(self.user_path.get('other_vehicle', {}) or {}).get('action')}")
        context_parts.append(f"  Collision point: {self.user_path.get('collision_point')}")
        
        return "\n".join(context_parts)
        

    def calculate_coverage_bounds(self):
        """Calculate the geographical bounds of all lanes"""
        if not self.lane_tree_data:
            return {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}
        
        all_x_coords = []
        all_y_coords = []
        
        for lane in self.lane_tree_data.get('lane_trees', []):
            points = lane.get('points', [])
            for point in points:
                try:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        all_x_coords.append(float(point[0]))
                        all_y_coords.append(float(point[1]))
                    elif isinstance(point, dict) and 'x' in point and 'y' in point:
                        all_x_coords.append(float(point['x']))
                        all_y_coords.append(float(point['y']))
                except (ValueError, TypeError, IndexError):
                    continue
        
        if not all_x_coords or not all_y_coords:
            return {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}
        
        return {
            "min_x": min(all_x_coords),
            "max_x": max(all_x_coords),
            "min_y": min(all_y_coords),
            "max_y": max(all_y_coords)
        }

    def estimate_intersection_center(self):
        """Estimate the center point of the intersection"""
        bounds = self.calculate_coverage_bounds()
        
        center_x = (bounds["min_x"] + bounds["max_x"]) / 2
        center_y = (bounds["min_y"] + bounds["max_y"]) / 2
        
        return {"x": center_x, "y": center_y}

    def calculate_approximate_length(self, points):
        """Calculate approximate length of a point sequence"""
        if len(points) < 2:
            return 0
        
        total_length = 0
        for i in range(len(points) - 1):
            try:
                if isinstance(points[i], (list, tuple)) and isinstance(points[i+1], (list, tuple)):
                    dx = float(points[i+1][0]) - float(points[i][0])
                    dy = float(points[i+1][1]) - float(points[i][1])
                    total_length += (dx*dx + dy*dy) ** 0.5
            except (ValueError, TypeError, IndexError):
                continue
        
        return total_length

    def calculate_general_direction(self, points):
        """Calculate general direction of a point sequence"""
        if len(points) < 2:
            return "unknown"
        
        try:
            start_point = points[0]
            end_point = points[-1]
            
            if isinstance(start_point, (list, tuple)) and isinstance(end_point, (list, tuple)):
                dx = float(end_point[0]) - float(start_point[0])
                dy = float(end_point[1]) - float(start_point[1])
                
                # Simple directional classification
                if abs(dx) > abs(dy):
                    return "east" if dx > 0 else "west"
                else:
                    return "north" if dy > 0 else "south"
        except (ValueError, TypeError, IndexError):
            pass
        
        return "unknown"


    def create_metadata_only_lane_tree(self):
        """Create metadata-only version without waypoints for LLM processing"""
        
        if not self.lane_tree_data:
            return {}
        
        metadata_only = {
            "lane_trees_metadata": [],
            "summary": {
                "total_lanes": len(self.lane_tree_data.get('lane_trees', [])),
                "coverage_bounds": self.calculate_coverage_bounds(),
                "intersection_center": self.estimate_intersection_center()
            }
        }
        
        for lane in self.lane_tree_data.get('lane_trees', []):
            # Extract only metadata, exclude waypoints
            lane_meta = {
                "lane_id": lane.get("lane_id"),
                "road_id": lane.get("road_id"),
                "direction": lane.get("direction"),
                "metadata": {
                    "parent_road_name": lane.get("metadata", {}).get("parent_road_name"),
                    "dir8": lane.get("metadata", {}).get("dir8"),
                    "traffic_direction": lane.get("metadata", {}).get("traffic_direction"),
                    "allowed_turns": lane.get("metadata", {}).get("allowed_turns", []),
                    "lane_type": lane.get("metadata", {}).get("lane_type"),
                    "simple_direction": lane.get("metadata", {}).get("simple_direction"),
                    "lateral_position_label": lane.get("metadata", {}).get("lateral_position_label"),
                    # Include narrative data for LLM understanding
                    "narrative_directional": lane.get("metadata", {}).get("narrative_directional", {}),
                    "can_turn_left": lane.get("metadata", {}).get("can_turn_left", False),
                    "can_turn_right": lane.get("metadata", {}).get("can_turn_right", False),
                    "can_go_straight": lane.get("metadata", {}).get("can_go_straight", False)
                },
                # Include geometric summary without full waypoints
                "geometry_summary": self.summarize_lane_geometry(lane),
                "connection_points": self.extract_connection_points(lane)
            }
            
            metadata_only["lane_trees_metadata"].append(lane_meta)
        
        return metadata_only

    def summarize_lane_geometry(self, lane):
        """Summarize lane geometry without full waypoints"""
        points = lane.get('points', [])
        if not points:
            return {"status": "no_geometry"}
        
        return {
            "start_point": points[0] if points else None,
            "end_point": points[-1] if len(points) > 1 else points[0],
            "total_waypoints": len(points),
            "approximate_length": self.calculate_approximate_length(points),
            "general_direction": self.calculate_general_direction(points)
        }

    def extract_connection_points(self, lane):
        """Extract key connection points for routing"""
        points = lane.get('points', [])
        if len(points) <= 2:
            return points
        
        # Return start, middle, and end points only
        mid_idx = len(points) // 2
        return [points[0], points[mid_idx], points[-1]]


    def _resolve_origin_lane_ids(self):
        """
        Populate user_path.origin_road_id and origin_lane_id deterministically when possible.
        Strategy:
        1) map origin_road name -> road_id
        2) collect lanes for that road_id
        3) filter by travel direction (derived from 'came from')
        4) filter by allowed_turns using intended maneuver
        5) pick first remaining; else leave unknown
        """
        origin_road_name = (self.user_path.get("origin_road") or "").strip().lower()
        origin_dir_std   = self.standardize_direction(self.user_path.get("origin_direction"))
        intended_std     = self.standardize_maneuver(self.user_path.get("intended_maneuver"))

        # (1) road_id from name (roads_meta)
        rid = self.road_name2id.get(origin_road_name)
        if rid is None:
            # also try lane index by parent_road_name
            # (If user provided a lane-friendly name that only exists in lane metadata)
            for (nm, _d), lanes in self.lanes_by_name_dir.items():
                if nm == origin_road_name and lanes:
                    rid = lanes[0].get("road_id")
                    break

        if rid is None:
            return  # keep unknown; UI may ask a follow-up

        self.user_path["origin_road_id"] = rid

        # (2) candidate lanes on that road
        candidates = list(self.lanes_by_road.get(rid, []))

        if not candidates:
            return

        # (3) filter by travel direction set (dir8)
        dirset = self._dir8_from_user_origin(origin_dir_std)
        def lane_dir(l):
            return (l.get("direction") or l.get("metadata", {}).get("dir8") or "").lower()
        c1 = [l for l in candidates if lane_dir(l) in dirset] or candidates

        # (4) filter by allowed_turns vs intended maneuver (if known)
        allowed = self._maneuver_to_allowed(intended_std)
        def lane_allowed(l):
            al = (l.get("metadata", {}).get("allowed_turns") or [])  # e.g. ["through","left","right"]
            return any(a in allowed for a in al)
        c2 = [l for l in c1 if lane_allowed(l)] or c1

        # (5) choose first deterministically; store lane_id
        chosen = c2[0]
        self.user_path["origin_lane_id"] = chosen.get("lane_id")





    def __init__(self, connection_id, bucket_name, infrastructure_data):
        self.connection_id = connection_id
        self.bucket_name = bucket_name
        self.infrastructure = infrastructure_data
        
        # ✅ KEEP JSON DATA ACCESSIBLE THROUGHOUT
        self.roads_data = infrastructure_data.get(f'{connection_id}_roads_metadata_only.json', {})
        self.network_data = infrastructure_data.get(f'{connection_id}_metadata_only_network.json', {})
        self.intersections_data = infrastructure_data.get(f'{connection_id}_intersections_metadata_only.json', {})
        self.lane_tree_data = infrastructure_data.get(f'{connection_id}_lane_tree_routes_enhanced.json', {})
        
        self.road_name2id, self.road_id2name = self._build_road_index()
        self.lane_by_id, self.lanes_by_road, self.lanes_by_name_dir = self._build_lane_index()

        # Investigation steps
        self.investigation_steps = [
            "approach_direction",
            "intended_action", 
            "lane_position",
            "traffic_signal",
            "other_vehicle_position",
            "other_vehicle_action",
            "collision_point",
            "accident_time"
        ]
        self.current_step = 0
        self.responses = {}
        
        self.user_path = {
            "origin_direction": None,
            "origin_road": None,
            "approach_lane": None,
            "intended_maneuver": None,
            "destination_road": None,
            "destination_direction": None,
            "traffic_conditions": {},
            "accident_details": {},
            "timeline": [],
            "other_vehicle": {},
            "collision_point": None,
            "accident_time": None
        }
        self.scene_understanding = None
        
    def analyze_infrastructure_scene(self):
        """Initial analysis of all infrastructure data"""
        logger.info("🔍 Starting comprehensive infrastructure analysis...")
        
        # Generate comprehensive scene understanding using all JSON data
        combined_data = {
            "intersections": self.intersections_data,
            "roads": self.roads_data,
            "network": self.network_data
        }
        
        prompt = f"""
        Analyze this complete traffic infrastructure data and provide a clear summary:
        
        {json.dumps(combined_data, indent=2)[:8000]}
        
        Please provide a comprehensive but concise summary covering:
        1. **Location Type**: What kind of intersection/road configuration is this?
        2. **Road Layout**: Describe the roads, their names, and orientations
        3. **Traffic Controls**: What traffic signals, signs, or controls are present?
        4. **Lane Structure**: How are the lanes configured for each direction?
        5. **Key Features**: Any important infrastructure elements for accident analysis
        
        Write this as if you're an expert traffic investigator explaining the scene.
        Keep it factual and clear.
        """
        
        self.scene_understanding = self.call_bedrock(prompt, max_tokens=1000)
        logger.info("✅ Infrastructure analysis complete")
        return self.scene_understanding
    
    def start_interactive_investigation(self):
        """Start the interactive conversation with user"""
        if not self.scene_understanding:
            return "Error: Scene analysis not completed"
        
        # Reset to first question
        self.current_step = 0
        
        # Get first question using JSON data
        first_question_data = self.get_next_question()
        
        return first_question_data["message"]

    def get_next_question(self):
        """Generate question using JSON infrastructure data"""
        if self.current_step >= len(self.investigation_steps):
            return self.generate_completion_summary()
        
        current_step_name = self.investigation_steps[self.current_step]
        
        # Generate question based on actual JSON data
        if current_step_name == "approach_direction":
            return self.generate_direction_question()
        elif current_step_name == "intended_action":
            return self.generate_action_question()
        elif current_step_name == "lane_position":
            return self.generate_lane_question()
        elif current_step_name == "traffic_signal":
            return self.generate_traffic_signal_question()
        elif current_step_name == "other_vehicle_position":
            return self.generate_other_vehicle_question()
        elif current_step_name == "other_vehicle_action":
            return self.generate_other_vehicle_action_question()
        elif current_step_name == "collision_point":
            return self.generate_collision_question()
        elif current_step_name == "accident_time":
            return self.generate_time_question()


    def generate_direction_question(self):
        """Generate direction question using ALL conversational identifiers"""
        
        # ✅ FIXED: Collect ALL conversational identifiers from ALL roads
        all_conversational_options = []
        
        for road in self.roads_data.get("roads_metadata", []):
            road_id = road.get("road_id")
            conv_identifiers = road.get("conversational_identifiers", [])
            
            # Add all conversational identifiers for this road
            for identifier in conv_identifiers:
                all_conversational_options.append({
                    "road_id": road_id,
                    "identifier": identifier
                })
        
        # Create natural options text using the ACTUAL conversational identifiers
        identifier_texts = [opt["identifier"] for opt in all_conversational_options]
        
        if len(identifier_texts) > 2:
            options_text = ", ".join(identifier_texts[:-1]) + f", or {identifier_texts[-1]}"
        elif len(identifier_texts) == 2:
            options_text = f"{identifier_texts[0]} or {identifier_texts[1]}"
        else:
            options_text = identifier_texts[0] if identifier_texts else "one of the roads"
        
        prompt = f"""
        Based on this intersection, generate a natural question asking which road the user came from.
        
        SCENE: {self.scene_understanding}
        
        Use these EXACT conversational road descriptions in your question:
        {identifier_texts}
        
        Generate a natural question like:
        "I can see this accident happened at this intersection. Which road were you traveling on - were you coming from {options_text}?"
        
        Use the exact conversational identifiers provided above - don't change the wording.
        """
        
        question = self.call_bedrock(prompt, max_tokens=400)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "approach_direction"
        }


    def generate_action_question(self):
        """Generate action question using roads/intersection JSON data"""
        prompt = f"""
        Based on this intersection data and the user's previous response, ask about their intended action:
        
        INTERSECTIONS DATA:
        {json.dumps(self.intersections_data, indent=2)[:3000]}
        
        ROADS DATA:
        {json.dumps(self.roads_data, indent=2)[:2000]}
        
        USER'S APPROACH DIRECTION:
        {self.responses.get('approach_direction', 'Not specified')}
        
        Generate a simple question asking if they were going straight, turning left, or turning right.
        Reference the actual road names from the JSON data for where they would be turning to if possible.
        
        Keep it simple: "Were you planning to go straight, turn left, or turn right?"
        """
        
        question = self.call_bedrock(prompt, max_tokens=300)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "intended_action"
        }

    def generate_lane_question(self):
        """Generate lane question using roads JSON data"""
        prompt = f"""
        Based on this road infrastructure data and previous responses, ask about which lane they were in:
        
        ROADS DATA:
        {json.dumps(self.roads_data, indent=2)[:3000]}
        
        USER'S DIRECTION AND ACTION:
        Direction: {self.responses.get('approach_direction', 'Not specified')}
        Intended Action: {self.responses.get('intended_action', 'Not specified')}
        
        Generate a simple question about which lane they were driving in.
        Reference the lane configuration from the JSON data if available.
        
        Keep it simple: "Which lane were you driving in?"
        """
        
        question = self.call_bedrock(prompt, max_tokens=200)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "lane_position"
        }

    def generate_traffic_signal_question(self):
        """Generate traffic signal question using intersections JSON data"""
        prompt = f"""
        Based on this intersection data, ask about traffic signals or controls:
        
        INTERSECTIONS DATA:
        {json.dumps(self.intersections_data, indent=2)[:3000]}
        
        Generate a question about traffic light color or traffic control.
        If the JSON data shows traffic signals, ask about the light color.
        If it shows stop signs, ask about stop signs.
        If no clear traffic control, ask generally.
        
        Keep it simple: "What color was the traffic light when you approached?" or "What traffic control was there (traffic light, stop sign, etc.)?"
        """
        
        question = self.call_bedrock(prompt, max_tokens=200)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "traffic_signal"
        }

    def generate_other_vehicle_question(self):
        """Generate other vehicle question with DISAMBIGUATION"""
        prompt = f"""
        Based on the intersection layout and user's path so far, ask about the other vehicle.
        
        INTERSECTION DATA:
        {json.dumps(self.intersections_data, indent=2)[:2000]}
        
        USER'S PATH SO FAR:
        {json.dumps(self.responses, indent=2)}
        
        Ask TWO things clearly:
        1. Where was the other vehicle coming from? (which direction/road)
        2. Was it on the SAME road as you, or a DIFFERENT road?
        
        Example good question:
        "Where was the other vehicle coming from? Was it:
        A) On the same road as you, but in the opposite lane/direction?
        B) Coming from a different road that meets at the intersection?"
        
        Keep it conversational but clear about the distinction.
        """
        
        question = self.call_bedrock(prompt, max_tokens=300)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "other_vehicle_position"
        }


    def generate_other_vehicle_action_question(self):
        """Generate other vehicle action question"""
        question = "What was the other vehicle doing when the accident happened - was it going straight, turning, or stopped?"
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "other_vehicle_action"
        }

    def generate_collision_question(self):
        """Generate collision point question using intersection data"""
        prompt = f"""
        Based on the intersection layout and both vehicles' paths, ask about collision point:
        
        INTERSECTION DATA:
        {json.dumps(self.intersections_data, indent=2)[:2000]}
        
        USER'S VEHICLE PATH:
        {json.dumps(self.responses, indent=2)}
        
        Ask where exactly the collision occurred within the intersection or road area.
        
        Keep it simple: "Where exactly did the collision happen?"
        """
        
        question = self.call_bedrock(prompt, max_tokens=200)
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "collision_point"
        }


    def generate_time_question(self):
        """Generate time question - FIXED to properly check existing data"""
        
        # FIXED: Check multiple sources for existing datetime info
        existing_datetime = None
        
        # Check if we have it in responses already
        if self.responses.get('accident_time'):
            existing_datetime = self.responses['accident_time']
        
        # Check if we have it from earlier collection
        elif hasattr(self, 'existing_collected_data'):
            if self.existing_collected_data.get('basic_q_0'):
                existing_datetime = self.existing_collected_data['basic_q_0']
            elif self.existing_collected_data.get('datetime'):
                existing_datetime = self.existing_collected_data['datetime']
        
        # Check user_path
        elif self.user_path.get("accident_time"):
            existing_datetime = self.user_path["accident_time"]
        
        if existing_datetime and existing_datetime != "unknown":
            # We already have this info, skip this question
            logger.info(f"⏭️ Skipping time question - already have datetime: {existing_datetime}")
            self.responses['accident_time'] = existing_datetime
            self.user_path["accident_time"] = existing_datetime
            
            # Move to completion
            self.current_step += 1
            return self.get_next_question()
        
        # Otherwise ask the question
        question = "What time did this accident occur? You can give me the approximate time if you're not sure of the exact time."
        
        return {
            "statusCode": 200,
            "message": question,
            "conversation_state": "accident_time"
        }


    def process_user_response(self, user_input, current_state):
        """FIXED: Ensure we process user's actual input"""
        
        logger.info(f"📥 Processing user response: '{user_input}' in state: '{current_state}'")
        
        # Store raw response
        self.responses[current_state] = user_input
        
        # Update path from response
        self.update_user_path_from_response(user_input, current_state)
        
        # Handle forced questions
        if getattr(self, "_next_forced", None):
            nxt = self._next_forced
            self._next_forced = None
            if "conversation_state" not in nxt:
                nxt["conversation_state"] = current_state
            return nxt
        
        # Advance to next question
        self.current_step += 1
        nxt = self.get_next_question()
        
        if not isinstance(nxt, dict):
            nxt = {"statusCode": 200, "message": str(nxt), "conversation_state": "unknown"}
        
        if "conversation_state" not in nxt:
            if 0 <= self.current_step < len(self.investigation_steps):
                nxt["conversation_state"] = self.investigation_steps[self.current_step]
            else:
                nxt["conversation_state"] = "investigation_complete"
        
        return nxt


    def _generate_clarification_question(self, unclear_response: str):
        """Generate better clarification with specific road suggestions"""
        
        # Find the most likely road based on partial matching
        likely_road_id = self._find_most_likely_road(unclear_response)
        
        if likely_road_id:
            # Store for confirmation handling
            self._last_suggested_road_id = likely_road_id
            
            # Get the conversational name
            road_name = "the road"
            for road in self.roads_data.get("roads_metadata", []):
                if str(road.get("road_id")) == likely_road_id:
                    conv_ids = road.get("conversational_identifiers", [])
                    road_name = conv_ids[0] if conv_ids else f"Road {road_id}"
                    break
            
            message = f"Based on your description '{unclear_response}', I think you mean {road_name}. Is that correct?"
        else:
            # List all options clearly
            road_options = []
            for road in self.roads_data.get("roads_metadata", []):
                conv_ids = road.get("conversational_identifiers", [])
                if conv_ids:
                    road_options.append(conv_ids[0])
            
            options_text = " or ".join(road_options)
            message = f"I'm not sure which road you mean. Could you specify: {options_text}?"
        
        return {
            "statusCode": 200,
            "message": message,
            "conversation_state": "approach_direction_clarification"
        }

    def _understand_road_from_natural_response(self, user_input: str) -> str:
        """FIXED: Better road ID validation and handling"""
        
        # ✅ CRITICAL: Get the ACTUAL road_ids from the loaded JSON data
        actual_road_ids = []
        for road in self.roads_data.get("roads_metadata", []):
            road_id = road.get("road_id")
            if road_id is not None:
                actual_road_ids.append(str(road_id))  # Convert to string for consistency
        
        logger.info(f"🔍 Actual road_ids available: {actual_road_ids}")
        
        # Build comprehensive matching data
        road_matching_options = []
        for road in self.roads_data.get("roads_metadata", []):
            road_id = str(road.get("road_id"))  # Convert to string
            conv_identifiers = road.get("conversational_identifiers", [])
            user_descriptions = road.get("user_likely_descriptions", [])
            narrative_data = road.get("narrative_directional", {})
            
            road_matching_options.append({
                "road_id": road_id,
                "conversational_identifiers": conv_identifiers,
                "user_descriptions": user_descriptions,
                "comes_from": narrative_data.get("where_it_comes_from", "")
            })
        
        # Create detailed matching text
        matching_text = ""
        for option in road_matching_options:
            matching_text += f"\nRoad ID: {option['road_id']}\n"
            matching_text += f"Conversational names: {', '.join(option['conversational_identifiers'][:2])}\n"
            matching_text += f"Users might say: {', '.join(option['user_descriptions'][:2])}\n"
            matching_text += f"Geographic area: {option['comes_from']}\n"
        
        prompt = f"""
        User response: "{user_input}"
        
        Available roads (EXACT road_ids you must return):
        {matching_text}
        
        CRITICAL: You must return EXACTLY one of these road_ids: {actual_road_ids}
        
        Match the user's response to the best road based on:
        1. Conversational identifiers
        2. User descriptions  
        3. Geographic references
        4. Directional clues
        
        Examples:
        - "from the bottom right" → match road with "southeast" area
        - "the diagonal street" → match road with "diagonal" in conversational names
        - "from upper left" → match road with "northwest" or "upper left" area
        
        Respond with ONLY the exact road_id from this list: {actual_road_ids}
        If unclear, respond with "unclear"
        """
        
        try:
            response = self.call_bedrock(prompt, max_tokens=20)
            result = response.strip()
            
            logger.info(f"🧠 LLM road matching: '{user_input}' → '{result}'")
            
            if result.lower() == "unclear":
                return None
                
            # ✅ STRICT VALIDATION: Only accept exact matches
            if result in actual_road_ids:
                logger.info(f"✅ Valid road_id matched: {result}")
                return result
            else:
                logger.warning(f"❌ LLM returned invalid road_id: '{result}'. Valid IDs: {actual_road_ids}")
                return None
                
        except Exception as e:
            logger.error(f"Failed road matching: {e}")
            return None

    def _is_confirmation(self, user_input: str) -> bool:
        """Check if user input is a confirmation"""
        confirmations = [
            "yes", "yeah", "yep", "correct", "right", "that's right", 
            "that's correct", "exactly", "yes that's right", "that's it",
            "확인", "맞습니다", "そうです", "对", "是的"  # Multi-language
        ]
        
        user_lower = user_input.lower().strip()
        return any(conf in user_lower for conf in confirmations)

    def _get_last_suggested_road_id(self) -> str:
        """Get the road_id that was suggested in the last clarification"""
        # This should be stored when generating clarification
        return getattr(self, '_last_suggested_road_id', None)

    def _store_road_selection(self, road_id: str):
        """Store the selected road with proper mapping"""
        self.user_path["origin_road_id"] = road_id
        
        # Find the conversational name for display
        for road in self.roads_data.get("roads_metadata", []):
            if str(road.get("road_id")) == road_id:
                conv_ids = road.get("conversational_identifiers", [])
                self.user_path["origin_road_display"] = conv_ids[0] if conv_ids else f"Road {road_id}"
                break
        
        logger.info(f"✅ Stored road selection: {road_id} -> {self.user_path.get('origin_road_display')}")



    def _parse_other_vehicle_position(self, user_input: str):
        """
        ENHANCED: Better scenario disambiguation
        """
        try:
            # Gather context
            user_road_id = self.user_path.get("origin_road_id")
            user_road_display = self.user_path.get("origin_road_display", "unknown")
            user_direction = self.user_path.get("origin_direction", "unknown")
            
            # Get available roads for context
            available_roads = []
            for road in self.roads_data.get("roads_metadata", []):
                road_id = road.get("road_id")
                conv_ids = road.get("conversational_identifiers", [])
                narrative = road.get("narrative_directional", {})
                
                available_roads.append({
                    "road_id": str(road_id),
                    "names": conv_ids[:2] if conv_ids else [f"Road_{road_id}"],
                    "comes_from": narrative.get("where_it_comes_from", ""),
                    "direction": road.get("direction", "")
                })
            
            prompt = f"""
            You are analyzing a traffic accident at an INTERSECTION. Parse where the OTHER vehicle was coming from.
            
            USER'S DESCRIPTION: "{user_input}"
            
            CONTEXT:
            - User's vehicle road_id: {user_road_id}
            - User's vehicle road name: {user_road_display}
            - User's vehicle direction: {user_direction}
            
            AVAILABLE ROADS AT THIS INTERSECTION:
            {json.dumps(available_roads, indent=2)}
            
            CRITICAL DISAMBIGUATION:
            
            Scenario A: DIFFERENT ROADS (Intersection Collision) ✅ MOST COMMON
            - Other vehicle is on a DIFFERENT road
            - They meet at the INTERSECTION
            - User says: "opposite direction", "facing me", "coming toward me", "from the other side"
            - This is an INTERSECTION collision, NOT a head-on collision
            - Example: User on Road 1 (SW), Other on Road 1 (NE) but they're DIFFERENT roads that happen to connect!
            
            Scenario B: SAME ROAD (Head-on Collision) ❌ RARE
            - Other vehicle is on the EXACT SAME road segment
            - They collide BEFORE reaching intersection
            - User says: "on my road", "in my lane", "head-on", "coming at me on the same street"
            - This is a HEAD-ON collision on the same road
            
            CRITICAL RULES:
            1. If user says "opposite direction" at an INTERSECTION → Scenario A (different roads)
            2. If user says "opposite direction on the SAME road" → Scenario B (same road)
            3. If user says "facing me" or "toward me" at intersection → Scenario A
            4. If collision is AT THE INTERSECTION → Scenario A (99% of cases)
            5. If collision is BEFORE the intersection → Scenario B
            
            ANALYSIS FOR THIS CASE:
            - User is on road {user_road_id} coming from {user_direction}
            - Other vehicle description: "{user_input}"
            - Is this an intersection collision (A) or head-on on same road (B)?
            
            OUTPUT FORMAT (JSON only):
            {{
                "scenario": "A",
                "road_id": "different_road_id",
                "road_name": "Road name",
                "direction": "northeast",
                "confidence": 0.95,
                "reasoning": "User said 'opposite direction' at an intersection, which means the other vehicle was on a different road approaching from the opposite side of the intersection (Scenario A), not a head-on collision on the same road."
            }}
            
            Default to Scenario A unless explicitly clear it's same-road head-on.
            """
            
            response = self.call_bedrock(prompt, max_tokens=400)
            
            # Parse LLM response
            result = self._extract_json_from_response(response)
            
            if result and result.get("confidence", 0) > 0.5:
                ov = self.user_path.setdefault("other_vehicle", {})
                ov["origin_road_id"] = result.get("road_id")
                ov["origin_road_display"] = result.get("road_name", f"Road_{result.get('road_id')}")
                ov["origin_direction"] = result.get("direction")
                ov["scenario"] = result.get("scenario", "A")  # Default to A
                
                logger.info(f"✅ LLM parsed other vehicle position:")
                logger.info(f"   Scenario: {result.get('scenario')} ({'Different roads' if result.get('scenario') == 'A' else 'Same road'})")
                logger.info(f"   Road: {result.get('road_name')} (ID: {result.get('road_id')})")
                logger.info(f"   Direction: {result.get('direction')}")
                logger.info(f"   Confidence: {result.get('confidence')}")
                logger.info(f"   Reasoning: {result.get('reasoning')}")
            else:
                logger.warning(f"⚠️ LLM uncertain about other vehicle position (confidence: {result.get('confidence', 0)})")
                logger.warning(f"   Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                
        except Exception as e:
            logger.error(f"❗ Failed to parse other vehicle position: {e}")

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extract JSON from LLM response, handling various formats
        """
        try:
            # Try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            logger.error(f"Could not extract JSON from: {response_text}")
            return None

    def _get_conversational_name(self, road_id) -> str:
        """Get the first conversational identifier for a road"""
        for road in self.roads_data.get("roads_metadata", []):
            if str(road.get("road_id")) == str(road_id):
                conv_ids = road.get("conversational_identifiers", [])
                return conv_ids[0] if conv_ids else None
        return None

    def _find_most_likely_road(self, user_input: str) -> str:
        """
        Find the most likely road based on partial matching with user input.
        Uses fuzzy matching against all road metadata.
        """
        user_lower = user_input.lower()
        
        # Score each road
        road_scores = []
        
        for road in self.roads_data.get("roads_metadata", []):
            road_id = str(road.get("road_id"))
            score = 0
            
            # Check conversational identifiers
            for identifier in road.get("conversational_identifiers", []):
                if identifier.lower() in user_lower:
                    score += len(identifier) * 2  # Higher weight for exact phrases
                else:
                    # Check word overlap
                    identifier_words = set(identifier.lower().split())
                    user_words = set(user_lower.split())
                    overlap = len(identifier_words & user_words)
                    score += overlap
            
            # Check user_likely_descriptions
            for description in road.get("user_likely_descriptions", []):
                if description.lower() in user_lower:
                    score += len(description) * 3  # Highest weight
                else:
                    description_words = set(description.lower().split())
                    user_words = set(user_lower.split())
                    overlap = len(description_words & user_words)
                    score += overlap * 1.5
            
            # Check narrative directional
            narrative = road.get("narrative_directional", {})
            comes_from = narrative.get("where_it_comes_from", "")
            if comes_from and comes_from.lower() in user_lower:
                score += len(comes_from) * 1.5
            
            if score > 0:
                road_scores.append((road_id, score))
        
        # Return road with highest score
        if road_scores:
            road_scores.sort(key=lambda x: x[1], reverse=True)
            best_road_id = road_scores[0][0]
            best_score = road_scores[0][1]
            logger.info(f"🎯 Best match: road_id={best_road_id}, score={best_score}")
            return best_road_id
        
        return None


    def _match_user_response_to_road_from_json(self, user_input: str) -> str:
        """
        Match user response directly to road using the metadata JSON.
        No need for separate mapping - the JSON IS the mapping!
        """
        user_lower = user_input.lower()
        
        best_match_road_id = None
        best_match_score = 0
        
        # Search through the metadata JSON
        for road in self.roads_data.get("roads_metadata", []):
            road_id = str(road.get("road_id"))
            score = 0
            
            # Check conversational identifiers (what we showed the user)
            conv_identifiers = road.get("conversational_identifiers", [])
            for identifier in conv_identifiers:
                identifier_lower = identifier.lower()
                
                # Exact substring match (highest score)
                if identifier_lower in user_lower:
                    score += len(identifier) * 10
                    logger.info(f"✅ Matched conversational ID: '{identifier}' → road_id={road_id}")
                else:
                    # Word overlap scoring
                    identifier_words = set(identifier_lower.split())
                    user_words = set(user_lower.split())
                    overlap = len(identifier_words & user_words)
                    if overlap > 0:
                        score += overlap * 5
                        logger.info(f"📝 Partial match (conv): {overlap} words → road_id={road_id}")
            
            # Check user_likely_descriptions (common phrases users say)
            user_descriptions = road.get("user_likely_descriptions", [])
            for description in user_descriptions:
                description_lower = description.lower()
                
                # Exact substring match
                if description_lower in user_lower:
                    score += len(description) * 15  # Even higher weight
                    logger.info(f"✅ Matched user description: '{description}' → road_id={road_id}")
                else:
                    # Word overlap
                    description_words = set(description_lower.split())
                    user_words = set(user_lower.split())
                    overlap = len(description_words & user_words)
                    if overlap > 0:
                        score += overlap * 7
                        logger.info(f"📝 Partial match (desc): {overlap} words → road_id={road_id}")
            
            # Check narrative directional info
            narrative = road.get("narrative_directional", {})
            comes_from = narrative.get("where_it_comes_from", "")
            if comes_from:
                comes_from_lower = comes_from.lower()
                
                if comes_from_lower in user_lower:
                    score += len(comes_from) * 8
                    logger.info(f"✅ Matched narrative: '{comes_from}' → road_id={road_id}")
                else:
                    # Word overlap
                    comes_words = set(comes_from_lower.split())
                    user_words = set(user_lower.split())
                    overlap = len(comes_words & user_words)
                    if overlap > 0:
                        score += overlap * 4
                        logger.info(f"📝 Partial match (narrative): {overlap} words → road_id={road_id}")
            
            # Update best match if this road scored higher
            if score > best_match_score:
                best_match_road_id = road_id
                best_match_score = score
        
        if best_match_road_id:
            logger.info(f"✅ Final match: '{user_input}' → road_id={best_match_road_id} (score: {best_match_score})")
            
            # ✅ VALIDATE: Check if this road actually has lanes matching the direction
            parsed_direction = self._parse_direction_from_natural_language(user_input)
            if parsed_direction and parsed_direction != "unknown":
                available_lanes = self._get_lanes_for_road(best_match_road_id)
                dirset = self._dir8_from_user_origin(parsed_direction)
                
                matching_lanes = [l for l in available_lanes 
                                if (l.get("direction") or "").lower() in dirset]
                
                if not matching_lanes:
                    logger.warning(f"⚠️ Road {best_match_road_id} has no {parsed_direction} lanes!")
                    logger.info(f"   Available lanes: {[l.get('direction') for l in available_lanes]}")
                    
                    # Try to find a better road with matching lanes
                    for road in self.roads_data.get("roads_metadata", []):
                        alt_road_id = str(road.get("road_id"))
                        if alt_road_id == best_match_road_id:
                            continue
                        
                        alt_lanes = self._get_lanes_for_road(alt_road_id)
                        alt_matching = [l for l in alt_lanes 
                                    if (l.get("direction") or "").lower() in dirset]
                        
                        if alt_matching:
                            logger.info(f"✅ Found better road with {parsed_direction} lanes: {alt_road_id}")
                            return alt_road_id
            
            return best_match_road_id
        
        logger.warning(f"⚠️ No match found for: '{user_input}'")
        return None


    def update_user_path_from_response(self, user_input, current_state):
        """Use JSON metadata directly - no separate mapping needed"""
        
        if current_state == "approach_direction":
            # Store the raw user input
            self.user_path["origin_direction"] = user_input
            
            # Parse direction
            parsed_direction = self._parse_direction_from_natural_language(user_input)
            if parsed_direction:
                self.user_path["origin_direction"] = parsed_direction
                logger.info(f"✅ Parsed direction: '{user_input}' → '{parsed_direction}'")
            
            # ✅ FIXED: Match directly from JSON metadata
            road_id = self._match_user_response_to_road_from_json(user_input)
            
            if road_id:
                self._store_road_selection(road_id)
                logger.info(f"✅ Road selection complete: {road_id}")
                return
            else:
                # Generate clarification
                self._next_forced = self._generate_clarification_question(user_input)
                return


    def _parse_direction_from_natural_language(self, user_input: str) -> str:
        """
        Parse natural language direction descriptions
        """
        try:
            prompt = f"""
            Parse this direction description: "{user_input}"
            
            Convert to ONE of these standard directions:
            north, south, east, west, northeast, northwest, southeast, southwest
            
            Examples:
            - "from bottom right" → "southeast"
            - "from upper left" → "northwest"
            - "from the left side" → "west"
            - "coming from north" → "north"
            
            Respond with ONLY the direction word. No other text.
            """
            
            response = self.call_bedrock(prompt, max_tokens=10)
            direction = response.strip().lower()
            
            valid_directions = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
            
            if direction in valid_directions:
                return direction
            else:
                logger.warning(f"⚠️ Invalid direction from LLM: '{direction}'")
                return "unknown"
                
        except Exception as e:
            logger.error(f"❗ Direction parsing failed: {e}")
            return "unknown"


    def extract_road_name_from_direction(self, direction_response):
        """Extract specific road name from direction response using JSON data"""
        prompt = f"""
        Based on this infrastructure data and user response, extract the specific road name:
        
        ROADS DATA:
        {json.dumps(self.roads_data, indent=2)[:2000]}
        
        INTERSECTIONS DATA:
        {json.dumps(self.intersections_data, indent=2)[:2000]}
        
        USER SAID: "{direction_response}"
        
        Extract the specific road name they were traveling on. Return just the road name or "unknown" if unclear.
        """
        
        try:
            road_name = self.call_bedrock(prompt, max_tokens=100)
            self.user_path["origin_road"] = road_name.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Failed to extract road name: {e}")
            self.user_path["origin_road"] = "unknown"

    def extract_destination_from_action(self, action_response):
        """Extract destination road from turning action"""
        if "turn" in action_response.lower():
            prompt = f"""
            Based on this infrastructure data and user's turning action, extract destination road:
            
            ROADS DATA:
            {json.dumps(self.roads_data, indent=2)[:2000]}
            
            USER'S ORIGIN: {self.user_path.get("origin_direction", "")}
            USER'S ACTION: "{action_response}"
            
            Extract the road name they would be turning onto. Return just the road name or "unknown" if unclear.
            """
            
            try:
                dest_road = self.call_bedrock(prompt, max_tokens=100)
                self.user_path["destination_road"] = dest_road.strip()
            except Exception as e:
                logger.warning(f"Failed to extract destination road: {e}")
                self.user_path["destination_road"] = "unknown"


    def generate_completion_summary(self):
        """Generate final summary and create route JSON - FIXED to use normalized datetime"""
        logger.info("🏁 Generating completion summary and route JSON")
        
        # ✅ NEW: Get normalized datetime (priority)
        accident_datetime = getattr(self, 'normalized_datetime', None)
        
        if not accident_datetime or accident_datetime == 'unknown':
            # Fallback to responses
            accident_datetime = self.responses.get('accident_time', 'unknown')
        
        logger.info(f"📅 Using datetime for summary: {accident_datetime}")
        
        # Generate analysis summary with EXPLICIT datetime instruction
        summary_prompt = f"""
        Generate a professional accident reconstruction summary based on these responses:
        
        CRITICAL: Use this EXACT date/time in your summary: {accident_datetime}
        DO NOT use relative terms like "yesterday" or "today". Use the exact date/time provided above.
        
        LOCATION INFRASTRUCTURE:
        {self.scene_understanding}
        
        INVESTIGATION RESULTS:
        {json.dumps(self.responses, indent=2)}
        
        USER PATH RECONSTRUCTION:
        {json.dumps(self.user_path, indent=2)}
        
        Provide:
        1. Brief summary of what happened
        2. Key factors that may have contributed
        3. Timeline of events leading to collision
        4. Traffic control analysis
        
        IMPORTANT: In the summary header, write:
        **Date/Time:** {accident_datetime}
        
        Write as a professional accident reconstruction report.
        """
        
        final_summary = self.call_bedrock(summary_prompt, max_tokens=1000)
        
        # ✅ CREATE ROUTE JSON
        route_json = self.create_route_json()
        
        # ✅ SAVE TO DYNAMODB
        self.save_to_dynamodb(route_json, final_summary)
        
        # Save animation data
        self.save_animation_data(route_json)

        # 🎬 Trigger SVG animation generation
        self.trigger_svg_generation(route_json)
        
        return {
            "statusCode": 200,
            "message": f"Thank you for providing all the details. Here's my analysis:\n\n{final_summary}\n\n✅ **Investigation Complete!** I've created a detailed route reconstruction and saved all the information for your claim processing.",
            "conversation_state": "investigation_complete",
            "animation_ready": True,
            "route_data": route_json,
            "final_analysis": final_summary
        }


    def standardize_direction(self, direction_input):
        """Convert any direction input to standard format"""
        if not direction_input:
            return "unknown"
        
        direction_map = {
            "north": "north", "n": "north", "northbound": "north",
            "south": "south", "s": "south", "southbound": "south", 
            "east": "east", "e": "east", "eastbound": "east",
            "west": "west", "w": "west", "westbound": "west",
            "northeast": "northeast", "ne": "northeast",
            "northwest": "northwest", "nw": "northwest",
            "southeast": "southeast", "se": "southeast", 
            "southwest": "southwest", "sw": "southwest"
        }
        
        direction_lower = direction_input.lower().strip()
        for key, standard in direction_map.items():
            if key in direction_lower:
                return standard
        
        return "unknown"

    def standardize_maneuver(self, maneuver_input):
        """Convert any maneuver input to standard format"""
        if not maneuver_input:
            return "unknown"
        
        maneuver_map = {
            "straight": ["straight", "through", "continue"],
            "left_turn": ["left", "turn left", "turning left"],
            "right_turn": ["right", "turn right", "turning right"],
            "u_turn": ["u-turn", "u turn", "uturn"],
            "stopped": ["stopped", "stop", "stationary"]
        }
        
        maneuver_lower = maneuver_input.lower().strip()
        for standard, variations in maneuver_map.items():
            if any(var in maneuver_lower for var in variations):
                return standard
        
        return "unknown"

    def standardize_lane(self, lane_input):
        """Convert any lane input to standard format"""
        if not lane_input:
            return "unknown"
        
        lane_map = {
            "left_lane": ["left", "left lane"],
            "right_lane": ["right", "right lane"], 
            "center_lane": ["center", "middle", "center lane"],
            "turning_lane": ["turning", "turn lane"],
            "shoulder": ["shoulder", "emergency"]
        }
        
        lane_lower = lane_input.lower().strip()
        for standard, variations in lane_map.items():
            if any(var in lane_lower for var in variations):
                return standard
        
        return "unknown"

    def determine_collision_type(self):
        """Determine collision type from vehicle paths"""
        user_maneuver = self.standardize_maneuver(self.user_path.get("intended_maneuver"))
        other_action = self.standardize_maneuver(self.user_path.get("other_vehicle", {}).get("action"))
        
        # Rule-based collision type determination
        if user_maneuver == "left_turn" and other_action == "straight":
            return "left_turn_collision"
        elif user_maneuver == "straight" and other_action == "left_turn":
            return "left_turn_collision"
        elif user_maneuver == "straight" and other_action == "straight":
            return "intersection_collision"
        else:
            return "unknown"

    def generate_animation_keyframes(self):
        """Generate standardized keyframes for animation"""
        return [
            {
                "time": 0,
                "user_vehicle": {"x": 0, "y": -100, "rotation": 0},
                "other_vehicle": {"x": -100, "y": 0, "rotation": 90}
            },
            {
                "time": 30,
                "user_vehicle": {"x": 0, "y": -20, "rotation": 0},
                "other_vehicle": {"x": -20, "y": 0, "rotation": 90}
            },
            {
                "time": 60,
                "user_vehicle": {"x": 0, "y": 0, "rotation": 0},
                "other_vehicle": {"x": 0, "y": 0, "rotation": 90}
            }
        ]



    def create_route_json(self):
        """Create comprehensive route JSON with FIXED STRUCTURE"""
        logger.info("🗺️ Creating standardized route JSON")

        # ✅ NEW: Check if already created
        if hasattr(self, '_route_json_created'):
            logger.warning("⚠️ Route JSON already created, skipping duplicate")
            return self._cached_route_json

        
        # ✅ FIXED STRUCTURE - NEVER CHANGES
        route_json = {
            "metadata": {
                "accident_id": self.connection_id,
                "created_timestamp": datetime.now().isoformat(),
                "schema_version": "1.0",
                "data_completeness": self.calculate_data_completeness()
            },
            
            "location": {
                "infrastructure": {
                    "intersection_type": self.extract_intersection_type(),
                    "road_names": self.extract_road_names(),
                    "traffic_controls": self.extract_traffic_controls(),
                    "lane_configuration": self.extract_lane_config(),
                    "coordinates": self.extract_location_coordinates()
                },
                "raw_data": {
                    "intersections_json": self.intersections_data,
                    "roads_json": self.roads_data,
                    "network_json": self.network_data
                }
            },
            
            "vehicles": {
                "user_vehicle": {
                    "path": {
                        "origin": {
                            "direction": self.standardize_direction(self.user_path.get("origin_direction")),
                            "road_name": self.user_path.get("origin_road_display", "unknown"),
                            "road_id": self.user_path.get("origin_road_id", -1),
                            "lane_id": "unknown",  # Will be determined in SVG process
                            "lane": "unknown"
                        },
                        "intended_destination": {
                            "maneuver": self.standardize_maneuver(self.user_path.get("intended_maneuver")),
                            "target_road": self.user_path.get("destination_road", "unknown")
                        },
                        "route_points": (
                            self._emit_route_points_from_lane(
                                self.user_path.get("origin_lane_id"),
                                label="user_origin_lane"
                            ) or self.generate_user_route_points()
                        )
                    },
                    "state_at_collision": {
                        "speed": "unknown",
                        "position": self.user_path.get("collision_point", "unknown"),
                        "action": self.user_path.get("intended_maneuver", "unknown")
                    }
                },

                "other_vehicle": {
                    "path": {
                        "origin": {
                            "direction": self.standardize_direction(
                                (self.user_path.get("other_vehicle", {}) or {}).get("origin_direction")
                                or (self.user_path.get("other_vehicle", {}) or {}).get("position")
                            ),
                            # ✅ FIXED: Handle other vehicle road data too
                            "road_name": (self.user_path.get("other_vehicle", {}) or {}).get("origin_road_display", "unknown"),
                            "road_id": (self.user_path.get("other_vehicle", {}) or {}).get("origin_road_id", -1),
                            "lane_id": "unknown",  # Will be determined in SVG process
                            "lane": "unknown"
                        },
                        "intended_destination": {
                            "maneuver": self.standardize_maneuver((self.user_path.get("other_vehicle", {}) or {}).get("action")),
                            "target_road": "unknown"
                        },
                        "route_points": (
                            self._emit_route_points_from_lane(
                                (self.user_path.get("other_vehicle", {}) or {}).get("origin_lane_id"),
                                label="other_origin_lane"
                            ) or self.generate_other_vehicle_route_points()
                        )
                    },
                    "state_at_collision": {
                        "speed": "unknown",
                        "position": (self.user_path.get("other_vehicle", {}) or {}).get("position", "unknown"),
                        "action": (self.user_path.get("other_vehicle", {}) or {}).get("action", "unknown")
                    }
                }
            },

            
            "collision": {
                "point": {
                    "description": self.user_path.get("collision_point", "unknown"),
                    "coordinates": self.estimate_collision_coordinates(),
                    "zone": self.determine_collision_zone()  # "intersection", "approach", "exit", etc.
                },
                "timestamp": {
                    "reported_time": self.user_path.get("accident_time", "unknown"),
                    "estimated_time": self.estimate_precise_time()
                },
                "type": self.determine_collision_type(),  # "t-bone", "rear-end", "head-on", etc.
                "severity": "unknown"  # Could be estimated
            },
            
            "traffic_conditions": {
                "signals": {
                    "user_vehicle_signal": self.extract_user_signal_state(),
                    "other_vehicle_signal": self.estimate_other_signal_state(),
                    "signal_type": self.extract_signal_type()  # "traffic_light", "stop_sign", "yield", "none"
                },
                "right_of_way": {
                    "user_vehicle_had_right": self.determine_right_of_way("user"),
                    "other_vehicle_had_right": self.determine_right_of_way("other"),
                    "right_of_way_rules": self.extract_right_of_way_rules()
                },
                "environmental": {
                    "weather": "unknown",
                    "visibility": "unknown", 
                    "road_conditions": "unknown"
                }
            },
            
            "timeline": {
                "events": [
                    {
                        "timestamp": "T-30s",
                        "event_type": "approach",
                        "description": f"User vehicle approaching from {self.user_path.get('origin_direction', 'unknown')}",
                        "vehicle": "user"
                    },
                    {
                        "timestamp": "T-10s", 
                        "event_type": "traffic_control_observation",
                        "description": f"Traffic signal: {self.responses.get('traffic_signal', 'unknown')}",
                        "vehicle": "user"
                    },
                    {
                        "timestamp": "T-5s",
                        "event_type": "other_vehicle_detected", 
                        "description": f"Other vehicle at: {self.responses.get('other_vehicle_position', 'unknown')}",
                        "vehicle": "other"
                    },
                    {
                        "timestamp": "T-0s",
                        "event_type": "collision",
                        "description": f"Collision at: {self.user_path.get('collision_point', 'unknown')}",
                        "vehicle": "both"
                    }
                ]
            },
            
            "analysis": {
                "investigation_responses": self.responses,
                "data_quality": {
                    "completeness_score": self.calculate_completeness_score(),
                    "confidence_level": self.calculate_confidence_level(),
                    "missing_data_points": self.identify_missing_data()
                },
                "fault_analysis": {
                    "preliminary_fault_assessment": "requires_further_analysis",
                    "contributing_factors": self.identify_contributing_factors(),
                    "traffic_violations": self.identify_potential_violations()
                }
            },
            
            "visualization_data": {
                "animation_keyframes": self.generate_animation_keyframes(),
                "camera_positions": self.generate_camera_positions(),
                "object_positions": self.generate_object_positions(),
                "highlight_zones": self.generate_highlight_zones()
            }
        }
        
        route_json = self.enhance_missing_values_with_llm(route_json)

        # ✅ Cache the result
        self._route_json_created = True
        self._cached_route_json = route_json

        return route_json



    def create_timeline_from_responses(self):
        """Create timeline from user responses"""
        timeline = []
        
        # Add timeline events based on responses
        if self.responses.get("approach_direction"):
            timeline.append({
                "event": "vehicle_approach",
                "description": f"User vehicle approaching from {self.responses['approach_direction']}",
                "timestamp": "T-30s"  # Estimated
            })
        
        if self.responses.get("traffic_signal"):
            timeline.append({
                "event": "traffic_signal_observed",
                "description": f"Traffic signal state: {self.responses['traffic_signal']}",
                "timestamp": "T-10s"
            })
        
        if self.responses.get("other_vehicle_position"):
            timeline.append({
                "event": "other_vehicle_noticed",
                "description": f"Other vehicle observed at: {self.responses['other_vehicle_position']}",
                "timestamp": "T-5s"
            })
        
        timeline.append({
            "event": "collision",
            "description": f"Collision occurred at: {self.responses.get('collision_point', 'intersection')}",
            "timestamp": "T-0s"
        })
        
        return timeline

    def estimate_other_vehicle_path(self):
        """Estimate other vehicle's path using infrastructure data"""
        if not self.responses.get("other_vehicle_position"):
            return "unknown"
        
        prompt = f"""
        Based on the infrastructure data and collision information, estimate the other vehicle's path:
        
        INFRASTRUCTURE:
        {json.dumps(self.intersections_data, indent=2)[:2000]}
        
        OTHER VEHICLE POSITION: {self.responses.get("other_vehicle_position")}
        OTHER VEHICLE ACTION: {self.responses.get("other_vehicle_action")}
        COLLISION POINT: {self.responses.get("collision_point")}
        
        Estimate the other vehicle's likely path and direction. Be brief.
        """
        
        try:
            return self.call_bedrock(prompt, max_tokens=200)
        except Exception as e:
            logger.warning(f"Failed to estimate other vehicle path: {e}")
            return "unknown"

    def estimate_collision_coordinates(self):
        """Estimate collision coordinates from infrastructure data"""
        # This would use the infrastructure JSON to estimate coordinates
        # For now, return a placeholder
        return {
            "estimated_lat": "from_infrastructure_data",
            "estimated_lon": "from_infrastructure_data",
            "confidence": "medium"
        }

    def generate_route_coordinates(self):
        """Generate route coordinates from infrastructure data"""
        # This would process the infrastructure JSON to create actual route coordinates
        # For now, return a structured placeholder
        return {
            "user_route_points": [],
            "other_vehicle_route_points": [],
            "collision_point_coordinates": {},
            "intersection_boundaries": {}
        }

    def save_to_dynamodb(self, route_json, final_summary):
        """Save route data and analysis to DynamoDB"""
        try:
            logger.info(f"💾 Saving route data to DynamoDB for {self.connection_id}")
            
            # Get existing item or create new one
            try:
                response = table.get_item(Key={"connection_id": self.connection_id})
                item = response.get("Item", {})
            except Exception as e:
                logger.warning(f"Could not get existing item: {e}")
                item = {}
            
            # Update with route data
            update_data = {
                "connection_id": self.connection_id,
                "route_analysis": {
                    "route_s3_key": f"animation-data/{self.connection_id}/complete_route_data.json",
                    "final_summary": final_summary,
                    "investigation_complete": True,
                    "completed_at": datetime.now().isoformat(),
                }                
            }
            
            # Merge with existing data
            item.update(update_data)
            
            # Convert to DynamoDB format
            #item_dynamodb = self.convert_to_dynamodb_format(item)
            
            # Save to DynamoDB
            table.put_item(Item=item)
            
            logger.info("✅ Route data saved to DynamoDB successfully")
            
        except Exception as e:
            logger.error(f"❗ Failed to save to DynamoDB: {e}")

    def convert_to_dynamodb_format(self, item):
        """Convert item to DynamoDB compatible format"""
        def convert_value(value):
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            elif isinstance(value, float):
                return Decimal(str(value))
            elif isinstance(value, int):
                return Decimal(str(value))
            else:
                return value
        
        return convert_value(item)


    def _infer_other_vehicle_road_from_context(self) -> str:
        """
        Use LLM to infer other vehicle's road from conversation context
        """
        try:
            # Prepare context
            user_road_id = self.user_path.get("origin_road_id")
            user_direction = self.user_path.get("origin_direction")
            other_position = (self.user_path.get("other_vehicle", {}) or {}).get("position", "")
            other_action = (self.user_path.get("other_vehicle", {}) or {}).get("action", "")
            
            # Get available roads
            available_roads = []
            for road in self.roads_data.get("roads_metadata", []):
                road_id = road.get("road_id")
                conv_ids = road.get("conversational_identifiers", [])
                available_roads.append({
                    "road_id": str(road_id),
                    "names": conv_ids[:2]
                })
            
            prompt = f"""
            Analyze this accident scenario and determine which road the OTHER vehicle was on:
            
            USER'S VEHICLE:
            - Road ID: {user_road_id}
            - Coming from: {user_direction}
            
            OTHER VEHICLE:
            - Position described as: "{other_position}"
            - Action: "{other_action}"
            
            AVAILABLE ROADS:
            {json.dumps(available_roads, indent=2)}
            
            Key clues:
            - "opposite direction of the same road" → SAME road_id as user
            - "from the left/right" → perpendicular road
            - "oncoming" → opposite direction, same road
            
            Respond with ONLY the road_id (as a number) that the other vehicle was on.
            If "opposite direction same road", return the user's road_id: {user_road_id}
            """
            
            response = self.call_bedrock(prompt, max_tokens=20)
            inferred_road_id = response.strip()
            
            # Validate it's a real road_id
            if inferred_road_id in [str(r["road_id"]) for r in available_roads]:
                logger.info(f"✅ LLM inferred other vehicle road: {inferred_road_id}")
                
                # Store it for future use
                self.user_path.setdefault("other_vehicle", {})
                self.user_path["other_vehicle"]["origin_road_id"] = inferred_road_id
                
                return inferred_road_id
            else:
                logger.warning(f"⚠️ LLM returned invalid road_id: {inferred_road_id}")
                return None
                
        except Exception as e:
            logger.error(f"❗ Failed to infer other vehicle road: {e}")
            return None

    def _find_perpendicular_road(self, user_road_id, other_direction: str) -> str:
        """
        Find a road that's perpendicular to user's road based on direction
        """
        user_direction = self.standardize_direction(self.user_path.get("origin_direction"))
        
        # Map directions to perpendicular directions
        perpendicular_map = {
            "north": ["east", "west"],
            "south": ["east", "west"],
            "east": ["north", "south"],
            "west": ["north", "south"],
            "northeast": ["northwest", "southeast"],
            "northwest": ["northeast", "southwest"],
            "southeast": ["southwest", "northeast"],
            "southwest": ["southeast", "northwest"]
        }
        
        target_directions = perpendicular_map.get(user_direction, [])
        
        logger.info(f"   🔍 Looking for perpendicular road: user={user_direction}, target={target_directions}")
        
        # Search for a road with matching direction
        for road_id, lanes in self.lanes_by_road.items():
            if str(road_id) == str(user_road_id):
                continue  # Skip user's road
            
            for lane in lanes:
                lane_dir = (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
                
                # Check if this lane's direction matches what we're looking for
                for target_dir in target_directions:
                    if target_dir in lane_dir or lane_dir in self._dir8_from_user_origin(other_direction):
                        logger.info(f"   ✅ Found perpendicular road: {road_id} (direction: {lane_dir})")
                        return road_id
        
        # Fallback: return any road that's not the user's
        for road_id in self.lanes_by_road.keys():
            if str(road_id) != str(user_road_id):
                logger.warning(f"   ⚠️ Using fallback road: {road_id}")
                return road_id
        
        logger.error(f"   ❌ Could not find perpendicular road!")
        return user_road_id  # Last resort

    def _get_lanes_for_road(self, road_id) -> list:
        """
        Type-safe lane lookup - tries all possible formats
        """
        if road_id is None:
            logger.warning("   ⚠️ road_id is None")
            return []
        
        # Try direct lookup
        lanes = self.lanes_by_road.get(road_id)
        if lanes:
            logger.info(f"   ✅ Found {len(lanes)} lanes (direct: {road_id})")
            return lanes
        
        # Try string conversion
        lanes = self.lanes_by_road.get(str(road_id))
        if lanes:
            logger.info(f"   ✅ Found {len(lanes)} lanes (string: '{road_id}')")
            return lanes
        
        # Try int conversion
        try:
            lanes = self.lanes_by_road.get(int(road_id))
            if lanes:
                logger.info(f"   ✅ Found {len(lanes)} lanes (int: {road_id})")
                return lanes
        except (ValueError, TypeError):
            pass
        
        # Try with underscores (e.g., "0_1")
        if isinstance(road_id, str) and '_' in road_id:
            lanes = self.lanes_by_road.get(road_id)
            if lanes:
                logger.info(f"   ✅ Found {len(lanes)} lanes (underscore: {road_id})")
                return lanes
        
        logger.warning(f"   ⚠️ No lanes found for road_id={road_id} (tried all formats)")
        logger.info(f"   Available road_ids: {list(self.lanes_by_road.keys())}")
        
        return []

    def _resolve_approach_lane_for_other(self) -> str:
        """
        COMPLETE FIX: Handles both scenarios with type-safe lookup and fallbacks
        """
        other_vehicle = self.user_path.get("other_vehicle", {}) or {}
        road_id = other_vehicle.get("origin_road_id")
        scenario = other_vehicle.get("scenario", "A")  # Default to different roads
        
        logger.info(f"🔍 Resolving other vehicle lane:")
        logger.info(f"   Scenario: {scenario} ({'Different roads' if scenario == 'A' else 'Same road'})")
        logger.info(f"   Road ID: {road_id} (type: {type(road_id).__name__})")
        
        user_road_id = self.user_path.get("origin_road_id")
        
        # ✅ Validate scenario against road_id
        if str(road_id) == str(user_road_id):
            if scenario == "A":
                logger.warning(f"   ⚠️ Scenario A but same road_id - correcting to Scenario B")
                scenario = "B"
                other_vehicle["scenario"] = "B"
        else:
            if scenario == "B":
                logger.warning(f"   ⚠️ Scenario B but different road_id - correcting to Scenario A")
                scenario = "A"
                other_vehicle["scenario"] = "A"
        
        # ========== SCENARIO B: Same Road, Opposite Lanes ==========
        if scenario == "B":
            logger.info("   🔄 Same road scenario - finding opposite lane")
            
            user_lane_id = self.user_path.get("origin_lane_id")
            
            # ✅ Type-safe lane lookup
            available_lanes = self._get_lanes_for_road(road_id)
            
            logger.info(f"   User lane: {user_lane_id}")
            logger.info(f"   Available lanes: {[l.get('lane_id') for l in available_lanes]}")
            
            if not available_lanes:
                logger.error(f"   ❌ No lanes found for road {road_id} - using fallback")
                return self._infer_conflicting_lane()
            
            # Find the lane that's NOT the user's lane
            for lane in available_lanes:
                lane_id = lane.get("lane_id")
                if lane_id and lane_id != user_lane_id:
                    logger.info(f"   ✅ Found opposite lane: {lane_id}")
                    return lane_id
            
            logger.warning(f"   ⚠️ Could not find opposite lane - using fallback")
            return self._infer_conflicting_lane()
        
        # ========== SCENARIO A: Different Roads ==========
        else:
            logger.info("   🔀 Different roads scenario")
            
            raw_direction = other_vehicle.get("origin_direction")
            origin_dir = self.standardize_direction(raw_direction)
            
            logger.info(f"   Raw direction: '{raw_direction}'")
            logger.info(f"   Standardized: '{origin_dir}'")
            
            # ✅ If road_id somehow matches user's, find different road
            if str(road_id) == str(user_road_id):
                logger.warning(f"   ⚠️ Road ID matches user - finding perpendicular road")
                road_id = self._find_perpendicular_road(user_road_id, origin_dir)
                logger.info(f"   ✅ Using road: {road_id}")
                other_vehicle["origin_road_id"] = road_id
            
            # ✅ Type-safe lane lookup
            available_lanes = self._get_lanes_for_road(road_id)
            
            if not available_lanes:
                logger.warning(f"   ⚠️ No lanes for road {road_id} - using fallback")
                return self._infer_conflicting_lane()
            
            # Filter by direction
            dirset = self._dir8_from_user_origin(origin_dir)
            logger.info(f"   Looking for directions: {dirset}")
            
            def lane_direction(lane):
                return (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
            
            candidates = [l for l in available_lanes if lane_direction(l) in dirset]
            
            logger.info(f"   Found {len(candidates)} candidates")
            
            if not candidates:
                logger.warning(f"   ⚠️ No direction match - using all lanes")
                candidates = available_lanes
            
            # Apply Japan LHT rules
            intended = self.standardize_maneuver(other_vehicle.get("action"))
            chosen_lane = self._apply_japan_lht_rules(candidates, intended)
            
            if chosen_lane:
                lane_id = chosen_lane.get("lane_id", "unknown")
                logger.info(f"   ✅ Resolved: {lane_id}")
                return lane_id
            else:
                logger.warning(f"   ⚠️ No lane chosen - using fallback")
                return self._infer_conflicting_lane()

    def _llm_infer_other_vehicle_road(self) -> str:
        """
        Use LLM to infer other vehicle's road from full conversation context
        """
        try:
            # Gather all context
            user_road_id = self.user_path.get("origin_road_id")
            user_direction = self.user_path.get("origin_direction")
            other_position = (self.user_path.get("other_vehicle", {}) or {}).get("position", "")
            other_action = (self.user_path.get("other_vehicle", {}) or {}).get("action", "")
            
            # Get available roads
            available_roads = []
            for road in self.roads_data.get("roads_metadata", []):
                road_id = road.get("road_id")
                conv_ids = road.get("conversational_identifiers", [])
                narrative = road.get("narrative_directional", {})
                
                available_roads.append({
                    "road_id": str(road_id),
                    "names": conv_ids[:2],
                    "comes_from": narrative.get("where_it_comes_from", ""),
                    "goes_to": narrative.get("where_it_goes_to", "")
                })
            
            prompt = f"""
            Determine which road the OTHER vehicle was traveling on in this accident:
            
            USER'S VEHICLE:
            - Road ID: {user_road_id}
            - Direction: {user_direction}
            
            OTHER VEHICLE:
            - Position: "{other_position}"
            - Action: "{other_action}"
            
            AVAILABLE ROADS:
            {json.dumps(available_roads, indent=2)}
            
            ANALYSIS RULES:
            1. If other vehicle was "opposite direction same road" → return user's road_id
            2. If other vehicle was "from the left/right" → find perpendicular road
            3. If other vehicle was "crossing" → find intersecting road
            4. Consider the geometry and typical traffic patterns
            
            OUTPUT (JSON only):
            {{
                "road_id": "4",
                "confidence": 0.9,
                "reasoning": "Brief explanation"
            }}
            """
            
            response = self.call_bedrock(prompt, max_tokens=200)
            result = self._extract_json_from_response(response)
            
            if result and result.get("confidence", 0) > 0.5:
                road_id = result.get("road_id")
                logger.info(f"✅ LLM inferred other vehicle road: {road_id}")
                logger.info(f"   Reasoning: {result.get('reasoning')}")
                return road_id
            else:
                logger.warning(f"⚠️ LLM uncertain about road (confidence: {result.get('confidence', 0)})")
                return None
                
        except Exception as e:
            logger.error(f"❗ LLM road inference failed: {e}")
            return None

    def _apply_japan_lht_rules(self, candidate_lanes: list, intended_maneuver: str) -> dict:
        """
        Apply Japan's Left-Hand Traffic rules to select appropriate lane
        
        Japan LHT Rules:
        - Drive on LEFT side of road
        - Right turns are PROTECTED (don't cross traffic)
        - Left turns are UNPROTECTED (cross oncoming traffic)
        - For straight: use left lane (default travel lane)
        - For left turn: use leftmost lane available
        - For right turn: use right lane
        """
        
        if not candidate_lanes:
            return None
        
        # Check if lanes have lateral position metadata
        def get_lateral_position(lane):
            meta = lane.get("metadata", {}) or {}
            return meta.get("lateral_position_label") or meta.get("lane_position") or "unknown"
        
        def get_allowed_turns(lane):
            meta = lane.get("metadata", {}) or {}
            return meta.get("allowed_turns") or []
        
        # Rule 1: Filter by allowed turns for the maneuver
        if intended_maneuver == "straight":
            # Prefer "through" lanes
            through_lanes = [l for l in candidate_lanes if "through" in get_allowed_turns(l)]
            if through_lanes:
                candidate_lanes = through_lanes
        
        elif intended_maneuver == "left_turn":
            # Prefer "left" lanes (leftmost, since Japan LHT)
            left_lanes = [l for l in candidate_lanes if "left" in get_allowed_turns(l)]
            if left_lanes:
                candidate_lanes = left_lanes
        
        elif intended_maneuver == "right_turn":
            # Prefer "right" lanes (rightmost)
            right_lanes = [l for l in candidate_lanes if "right" in get_allowed_turns(l)]
            if right_lanes:
                candidate_lanes = right_lanes
        
        # Rule 2: If still multiple candidates, prefer by lateral position
        # In Japan LHT: "left" lane is default, "right" lane is passing/turning
        
        if len(candidate_lanes) > 1:
            # For straight: prefer left lane
            if intended_maneuver == "straight":
                left_positioned = [l for l in candidate_lanes if "left" in get_lateral_position(l).lower()]
                if left_positioned:
                    return left_positioned[0]
            
            # For left turn: prefer leftmost
            elif intended_maneuver == "left_turn":
                left_positioned = [l for l in candidate_lanes if "left" in get_lateral_position(l).lower()]
                if left_positioned:
                    return left_positioned[0]
            
            # For right turn: prefer rightmost
            elif intended_maneuver == "right_turn":
                right_positioned = [l for l in candidate_lanes if "right" in get_lateral_position(l).lower()]
                if right_positioned:
                    return right_positioned[0]
        
        # Rule 3: Fallback - return first candidate
        return candidate_lanes[0]


    def _infer_conflicting_lane(self) -> str:
        """
        LLM-powered inference of conflicting lane when road data is incomplete
        """
        try:
            user_road_id = self.user_path.get("origin_road_id")
            user_direction = self.user_path.get("origin_direction")
            other_position = (self.user_path.get("other_vehicle", {}) or {}).get("position", "")
            other_action = (self.user_path.get("other_vehicle", {}) or {}).get("action", "")
            
            # Get all available lanes
            all_lanes = []
            for road_id, lanes in self.lanes_by_road.items():
                for lane in lanes:
                    lane_dir = (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
                    all_lanes.append({
                        "lane_id": lane.get("lane_id"),
                        "road_id": road_id,
                        "direction": lane_dir,
                        "turns": lane.get("metadata", {}).get("allowed_turns", [])
                    })
            
            prompt = f"""
            Find the most likely lane for the OTHER vehicle that would create a collision:
            
            USER'S VEHICLE:
            - Road: {user_road_id}
            - Direction: {user_direction}
            
            OTHER VEHICLE:
            - Position: "{other_position}"
            - Action: "{other_action}"
            
            AVAILABLE LANES:
            {json.dumps(all_lanes[:20], indent=2)}
            
            Find a lane that:
            1. Would intersect with user's path
            2. Matches the other vehicle's described position/action
            3. Creates a realistic collision scenario
            
            OUTPUT (JSON only):
            {{
                "lane_id": "road_X_Y_lane",
                "confidence": 0.8,
                "reasoning": "This lane would create collision because..."
            }}
            """
            
            response = self.call_bedrock(prompt, max_tokens=200)
            result = self._extract_json_from_response(response)
            
            if result and result.get("confidence", 0) > 0.6:
                lane_id = result.get("lane_id")
                logger.info(f"✅ LLM inferred conflicting lane: {lane_id}")
                logger.info(f"   Reasoning: {result.get('reasoning')}")
                return lane_id
            else:
                logger.warning("⚠️ Could not infer conflicting lane with confidence")
                return "unknown"
                
        except Exception as e:
            logger.error(f"❗ Conflicting lane inference failed: {e}")
            return "unknown"


    def _resolve_approach_lane_for_user(self) -> str:
        """Resolve user's approach lane with ENHANCED DEBUGGING"""
        road_id = self.user_path.get("origin_road_id")
        origin_dir = self.standardize_direction(self.user_path.get("origin_direction"))
        intended = self.standardize_maneuver(self.user_path.get("intended_maneuver"))
        
        logger.info(f"🔍 DEBUG: _resolve_approach_lane_for_user")
        logger.info(f"🔍 DEBUG: road_id={road_id}, origin_dir='{origin_dir}', intended='{intended}'")
        
        if not road_id:
            logger.warning("⚠️ No road_id for user, cannot resolve lane")
            return "unknown"
        
        # 🔍 DEBUG: Check what lanes we have for this road
        available_lanes = self.lanes_by_road.get(road_id, [])
        logger.info(f"🔍 DEBUG: Road {road_id} has {len(available_lanes)} lanes")
        
        if not available_lanes:
            logger.warning(f"⚠️ No lanes found for road {road_id}")
            # 🔍 DEBUG: Show what roads DO have lanes
            logger.info(f"🔍 DEBUG: Roads with lanes: {list(self.lanes_by_road.keys())}")
            
            # 🔧 FIX: Try to find lanes with matching road_id using string comparison
            for road_str, lanes_list in self.lanes_by_road.items():
                if str(road_str) == str(road_id):
                    logger.info(f"✅ DEBUG: Found road {road_id} using string match!")
                    available_lanes = lanes_list
                    break
            
            if not available_lanes:
                return "unknown"
        
        # Filter by traffic direction (toward intersection)
        dirset = self._dir8_from_user_origin(origin_dir)
        logger.info(f"🔍 DEBUG: Looking for lane directions in: {dirset}")
        
        def lane_direction(lane):
            return (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
        
        candidates = [l for l in available_lanes if lane_direction(l) in dirset]
        logger.info(f"🔍 DEBUG: Found {len(candidates)} candidate lanes matching direction")
        
        if not candidates:
            logger.warning(f"⚠️ No lanes match direction {dirset} for road {road_id}")
            # 🔍 DEBUG: Show what directions ARE available
            available_directions = [lane_direction(l) for l in available_lanes]
            logger.info(f"🔍 DEBUG: Available lane directions: {available_directions}")
            candidates = available_lanes  # Fallback to all lanes
        
        # Apply Japan LHT rules (vehicles drive on left)
        chosen_lane = self._apply_japan_lht_rules(candidates, intended)
        
        lane_id = chosen_lane.get("lane_id", "unknown") if chosen_lane else "unknown"
        
        logger.info(f"✅ User lane resolved: {lane_id} (road={road_id}, dir={origin_dir}, maneuver={intended})")
        
        return lane_id

    def save_animation_data(self, route_json):
        """Save both full data and metadata-only version with FIXED STRUCTURE"""
        
        # Save full animation data (existing code)
        animation_data = {
            "user_path": self.user_path,
            "infrastructure": self.scene_understanding,
            "route_json": route_json,
            "timeline": self.user_path.get("timeline", []),
            "accident_point": self.user_path.get("collision_point"),
            "created_at": datetime.now().isoformat()
        }
        
        animation_key = f"animation-data/{self.connection_id}/complete_route_data.json"
        s3.put_object(
            Bucket=self.bucket_name,
            Key=animation_key,
            Body=json.dumps(animation_data, indent=2),
            ContentType='application/json'
        )
        
        # ✅ NEW: Create metadata-only version with STABLE TOP-LEVEL STRUCTURE
        metadata_only_lane_tree = self.create_metadata_only_lane_tree()
        
        # 🔧 FIX: Pre-compute lane IDs using Japan LHT rules
        user_lane_id = self._resolve_approach_lane_for_user()
        other_lane_id = self._resolve_approach_lane_for_other()
        
        logger.info(f"✅ Resolved lanes: user={user_lane_id}, other={other_lane_id}")
        
        llm_data = {
            # 🆕 ADD: Simple top-level keys for SVG generator
            "user_path": {
                "approach_lane": user_lane_id,  # ← This is what SVG generator needs
                "lane_id": user_lane_id,
                "origin_road_id": self.user_path.get("origin_road_id"),
                "origin_direction": self.standardize_direction(self.user_path.get("origin_direction")),
                "intended_maneuver": self.standardize_maneuver(self.user_path.get("intended_maneuver")),
            },
            
            "other_vehicle_path": {
                "approach_lane": other_lane_id,
                "lane_id": other_lane_id,
                "origin_road_id": (self.user_path.get("other_vehicle", {}) or {}).get("origin_road_id"),
                "origin_direction": self.standardize_direction(
                    (self.user_path.get("other_vehicle", {}) or {}).get("origin_direction") or
                    (self.user_path.get("other_vehicle", {}) or {}).get("position")
                ),
                "intended_maneuver": self.standardize_maneuver(
                    (self.user_path.get("other_vehicle", {}) or {}).get("action")
                ),
            },
            
            # Keep existing structure for compatibility
            "route_analysis": route_json,
            "lane_tree_metadata": metadata_only_lane_tree,
            
            # ✅ FIXED: Complete infrastructure_summary
            "infrastructure_summary": {
                "roads": self.roads_data,
                "intersections": self.intersections_data,
                "network": self.network_data
            },
            
            # ✅ FIXED: Complete accident_scenario
            "accident_scenario": {
                "user_vehicle": route_json.get("vehicles", {}).get("user_vehicle", {}),
                "other_vehicle": route_json.get("vehicles", {}).get("other_vehicle", {}),
                "collision": route_json.get("collision", {})
            }
        }
        
        llm_key = f"animation-data/{self.connection_id}/llm_ready_data.json"
        s3.put_object(
            Bucket=self.bucket_name,
            Key=llm_key,
            Body=json.dumps(llm_data, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"💾 Saved LLM-ready data with stable structure: {llm_key}")
        logger.info(f"📍 User lane: {user_lane_id}")
        logger.info(f"📍 Other lane: {other_lane_id}")


    def call_bedrock(self, prompt, max_tokens=500):
        """Helper to call Bedrock"""
        try:
            response = bedrock.invoke_model(
                modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            return json.loads(response["body"].read())["content"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"❗ Bedrock call failed: {e}")
            return f"Error: {str(e)}"

    def calculate_data_completeness(self):
        """Calculate how complete the collected data is"""
        total_fields = len(self.investigation_steps)
        completed_fields = len([k for k, v in self.responses.items() if v and str(v).strip().lower() != "unknown"])
        
        return {
            "total_questions": total_fields,
            "answered_questions": completed_fields,
            "completion_percentage": (completed_fields / total_fields) * 100 if total_fields > 0 else 0,
            "missing_responses": [step for step in self.investigation_steps if step not in self.responses or not self.responses[step]]
        }

    def extract_intersection_type(self):
        """Extract intersection type from infrastructure data"""
        try:
            # Analyze intersection data to determine type
            if self.intersections_data:
                prompt = f"""
                Based on this intersection data, what type of intersection is this?
                
                {json.dumps(self.intersections_data, indent=2)[:1000]}
                
                Return one of: "4-way_intersection", "3-way_intersection", "T-intersection", "roundabout", "highway_interchange", "simple_road", "unknown"
                """
                return self.call_bedrock(prompt, max_tokens=50).strip()
            return "unknown"
        except Exception as e:
            logger.warning(f"Failed to extract intersection type: {e}")
            return "unknown"

    def extract_road_names(self):
        """Extract road names from infrastructure data"""
        try:
            road_names = []
            if self.roads_data:
                # Try to extract road names from the roads data
                prompt = f"""
                Extract all road names from this data:
                
                {json.dumps(self.roads_data, indent=2)[:2000]}
                
                Return a simple list of road names, one per line. If no names found, return "unknown".
                """
                result = self.call_bedrock(prompt, max_tokens=200)
                road_names = [name.strip() for name in result.split('\n') if name.strip()]
            
            return road_names if road_names else ["unknown"]
        except Exception as e:
            logger.warning(f"Failed to extract road names: {e}")
            return ["unknown"]

    def extract_traffic_controls(self):
        """Extract traffic control information"""
        try:
            if self.intersections_data:
                prompt = f"""
                What traffic controls are present at this location?
                
                {json.dumps(self.intersections_data, indent=2)[:1500]}
                
                Return a simple list: traffic_lights, stop_signs, yield_signs, or none
                """
                return self.call_bedrock(prompt, max_tokens=100).strip().split(', ')
            return ["unknown"]
        except Exception as e:
            logger.warning(f"Failed to extract traffic controls: {e}")
            return ["unknown"]

    def extract_lane_config(self):
        """Extract lane configuration"""
        try:
            if self.roads_data:
                prompt = f"""
                Describe the lane configuration at this location:
                
                {json.dumps(self.roads_data, indent=2)[:1500]}
                
                How many lanes in each direction? Return brief description.
                """
                return self.call_bedrock(prompt, max_tokens=150).strip()
            return "unknown"
        except Exception as e:
            logger.warning(f"Failed to extract lane config: {e}")
            return "unknown"

    def extract_location_coordinates(self):
        """Extract location coordinates from infrastructure data"""
        try:
            # Try to find coordinates in the infrastructure data
            coords = {"lat": "unknown", "lon": "unknown"}
            
            # Check all infrastructure data for coordinate information
            all_data = {**self.intersections_data, **self.roads_data, **self.network_data}
            
            # Simple search for coordinate-like data
            for key, value in all_data.items():
                if isinstance(value, (int, float)) and "lat" in str(key).lower():
                    coords["lat"] = value
                elif isinstance(value, (int, float)) and "lon" in str(key).lower():
                    coords["lon"] = value
            
            return coords
        except Exception as e:
            logger.warning(f"Failed to extract coordinates: {e}")
            return {"lat": "unknown", "lon": "unknown"}

    def generate_user_route_points(self):
        """Generate route points for user vehicle"""
        return [
            {
                "point_type": "origin",
                "description": f"Starting from {self.user_path.get('origin_direction', 'unknown')} direction",
                "coordinates": {"x": 0, "y": -100}  # Placeholder coordinates
            },
            {
                "point_type": "approach",
                "description": f"Approaching intersection in {self.user_path.get('approach_lane', 'unknown')} lane",
                "coordinates": {"x": 0, "y": -50}
            },
            {
                "point_type": "collision",
                "description": f"Collision point: {self.user_path.get('collision_point', 'unknown')}",
                "coordinates": {"x": 0, "y": 0}
            }
        ]

    def generate_other_vehicle_route_points(self):
        """Generate route points for other vehicle"""
        return [
            {
                "point_type": "origin", 
                "description": f"Other vehicle from {self.user_path.get('other_vehicle', {}).get('position', 'unknown')}",
                "coordinates": {"x": -100, "y": 0}  # Placeholder
            },
            {
                "point_type": "collision",
                "description": f"Other vehicle action: {self.user_path.get('other_vehicle', {}).get('action', 'unknown')}",
                "coordinates": {"x": 0, "y": 0}
            }
        ]

    def estimate_other_vehicle_origin(self):
        """Estimate where other vehicle came from"""
        other_pos = self.user_path.get("other_vehicle", {}).get("position", "")
        if "left" in other_pos.lower():
            return "left"
        elif "right" in other_pos.lower():
            return "right"
        elif "opposite" in other_pos.lower() or "front" in other_pos.lower():
            return "opposite"
        return "unknown"

    def determine_collision_zone(self):
        """Determine which zone collision occurred in"""
        collision_point = self.user_path.get("collision_point", "").lower()
        if "intersection" in collision_point or "middle" in collision_point:
            return "intersection_center"
        elif "approach" in collision_point:
            return "intersection_approach"
        elif "exit" in collision_point:
            return "intersection_exit"
        return "unknown"

    def estimate_precise_time(self):
        """Estimate more precise time from user input"""
        accident_time = self.user_path.get("accident_time", "")
        if accident_time and accident_time != "unknown":
            return f"Estimated: {accident_time}"
        return "unknown"

    def extract_user_signal_state(self):
        """Extract what signal user vehicle had"""
        return self.responses.get("traffic_signal", "unknown")

    def extract_signal_type(self):
        """Extract type of traffic control"""
        signal_response = self.responses.get("traffic_signal", "").lower()
        if "light" in signal_response or "red" in signal_response or "green" in signal_response or "yellow" in signal_response:
            return "traffic_light"
        elif "stop" in signal_response:
            return "stop_sign"
        elif "yield" in signal_response:
            return "yield_sign"
        return "unknown"

    def estimate_other_signal_state(self):
        """Estimate what signal other vehicle had"""
        user_signal = self.extract_user_signal_state().lower()
        if "green" in user_signal:
            return "likely_red_or_stop"
        elif "red" in user_signal:
            return "likely_green"
        return "unknown"

    def determine_right_of_way(self, vehicle):
        """Determine if vehicle had right of way"""
        signal_state = self.extract_user_signal_state().lower()
        user_action = self.user_path.get("intended_maneuver", "").lower()
        
        if vehicle == "user":
            if "green" in signal_state and "straight" in user_action:
                return True
            elif "red" in signal_state:
                return False
            return "unclear"
        else:  # other vehicle
            # Opposite of user's right of way
            user_right = self.determine_right_of_way("user")
            if user_right is True:
                return False
            elif user_right is False:
                return True
            return "unclear"

    def extract_right_of_way_rules(self):
        """Extract applicable right of way rules"""
        rules = []
        signal_type = self.extract_signal_type()
        user_action = self.user_path.get("intended_maneuver", "").lower()
        
        if signal_type == "traffic_light":
            rules.append("traffic_light_control")
            if "turn" in user_action:
                rules.append("turning_yields_to_oncoming")
        elif signal_type == "stop_sign":
            rules.append("stop_sign_control")
            rules.append("first_to_stop_has_right_of_way")
        
        return rules

    def calculate_completeness_score(self):
        """Calculate data completeness score"""
        completion = self.calculate_data_completeness()
        return completion["completion_percentage"]

    def calculate_confidence_level(self):
        """Calculate confidence level of analysis"""
        completeness = self.calculate_completeness_score()
        if completeness >= 90:
            return "high"
        elif completeness >= 70:
            return "medium"
        elif completeness >= 50:
            return "low"
        return "very_low"

    def identify_missing_data(self):
        """Identify what data is missing"""
        completion = self.calculate_data_completeness()
        return completion["missing_responses"]

    def identify_contributing_factors(self):
        """Identify factors that may have contributed to accident"""
        factors = []
        
        # Check for potential factors based on responses
        signal_state = self.extract_user_signal_state().lower()
        if "red" in signal_state:
            factors.append("possible_red_light_violation")
        
        user_action = self.user_path.get("intended_maneuver", "").lower()
        if "turn" in user_action:
            factors.append("turning_maneuver")
        
        # Check intersection type
        intersection_type = self.extract_intersection_type()
        if "4-way" in intersection_type:
            factors.append("complex_intersection")
        
        return factors

    def identify_potential_violations(self):
        """Identify potential traffic violations"""
        violations = []
        
        signal_state = self.extract_user_signal_state().lower()
        if "red" in signal_state:
            violations.append("red_light_violation")
        
        # Add more violation checks based on responses
        return violations

    def generate_camera_positions(self):
        """Generate camera positions for visualization"""
        return [
            {"name": "overhead", "position": {"x": 0, "y": 0, "z": 100}, "target": {"x": 0, "y": 0, "z": 0}},
            {"name": "user_vehicle_perspective", "position": {"x": 0, "y": -80, "z": 5}, "target": {"x": 0, "y": 0, "z": 0}},
            {"name": "intersection_corner", "position": {"x": -50, "y": -50, "z": 20}, "target": {"x": 0, "y": 0, "z": 0}}
        ]

    def generate_object_positions(self):
        """Generate positions for objects in visualization"""
        return {
            "traffic_lights": [{"position": {"x": 10, "y": 10, "z": 5}, "type": "traffic_light"}],
            "stop_signs": [],
            "lane_markings": [{"start": {"x": -20, "y": 0}, "end": {"x": 20, "y": 0}}],
            "crosswalks": []
        }

    def generate_highlight_zones(self):
        """Generate zones to highlight in visualization"""
        return [
            {
                "name": "collision_zone",
                "type": "circle",
                "center": {"x": 0, "y": 0},
                "radius": 5,
                "color": "red",
                "opacity": 0.3
            },
            {
                "name": "approach_zone", 
                "type": "rectangle",
                "bounds": {"x1": -5, "y1": -50, "x2": 5, "y2": -5},
                "color": "yellow",
                "opacity": 0.2
            }
        ]

    def _build_road_index(self):
        name2id, id2name = {}, {}
        for r in (self.roads_data or {}).get("roads_metadata", []):
            rid = r.get("road_id")
            nm  = (r.get("name") or r.get("display_name") or "").strip()
            if nm and rid is not None:
                name2id[nm.lower()] = rid
                id2name[rid] = nm
        return name2id, id2name

    def _dir8_label(self, d: str) -> str:
        d = (d or "unknown").lower()
        if d in ("n","north"): return "NB"
        if d in ("s","south"): return "SB"
        if d in ("e","east"):  return "EB"
        if d in ("w","west"):  return "WB"
        if "northwest" in d or d=="nw": return "NW"
        if "northeast" in d or d=="ne": return "NE"
        if "southwest" in d or d=="sw": return "SW"
        if "southeast" in d or d=="se": return "SE"
        return d.upper()


    def _build_lane_index(self):
        """Build lane indexes with ENHANCED DEBUGGING"""
        by_lane_id, by_road, by_name_dir = {}, {}, {}
        
        logger.info("🔍 DEBUG: Building lane index...")
        
        lane_trees = (self.lane_tree_data or {}).get("lane_trees", [])
        logger.info(f"🔍 DEBUG: Found {len(lane_trees)} lanes in lane_tree_data")
        
        for i, lane in enumerate(lane_trees):
            lid = lane.get("lane_id")
            rid = lane.get("road_id")
            meta = lane.get("metadata", {}) or {}
            name = (meta.get("parent_road_name") or meta.get("road_name") or "").strip()
            direction = (lane.get("direction") or meta.get("dir8") or meta.get("traffic_direction") or "unknown").lower()
            
            # 🔍 DEBUG: Log first few lanes in detail
            if i < 5:
                logger.info(f"🔍 DEBUG: Lane {i}: lane_id='{lid}', road_id={rid}, direction='{direction}'")
                logger.info(f"🔍 DEBUG: Lane {i} metadata keys: {list(meta.keys())}")
            
            if lid:
                by_lane_id[lid] = lane
            if rid is not None:
                by_road.setdefault(rid, []).append(lane)
                logger.info(f"🔍 DEBUG: Added lane {lid} to road {rid}")
            if name:
                by_name_dir.setdefault((name.lower(), direction), []).append(lane)
        
        # 🔍 DEBUG: Show final road->lane mapping
        logger.info(f"🔍 DEBUG: Final road->lane mapping:")
        for road_id, lanes_list in by_road.items():
            logger.info(f"🔍 DEBUG: Road {road_id}: {len(lanes_list)} lanes")
            for lane in lanes_list[:2]:  # Show first 2 lanes per road
                logger.info(f"🔍 DEBUG:   - {lane.get('lane_id')}")
        
        # Debug: Show all available roads
        logger.info(f"🔍 DEBUG: Available road IDs: {list(by_road.keys())}")
        logger.info(f"🔍 DEBUG: Total roads in index: {len(by_road)}")
        
        return by_lane_id, by_road, by_name_dir


    def _alias_for_road(self, road_id: int, dir8: str = None) -> str:
        """FIXED: Better road naming with actual direction analysis"""
        base = self.road_id2name.get(road_id, f"Road_{road_id}")
        
        # ✅ GET ACTUAL DIRECTION FROM LANE DATA
        actual_directions = []
        for lane in self.lanes_by_road.get(road_id, []):
            lane_dir = (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
            if lane_dir and lane_dir not in actual_directions:
                actual_directions.append(lane_dir)
        
        if actual_directions:
            # Show all directions this road serves
            dir_labels = [self._dir8_label(d) for d in actual_directions[:2]]  # Max 2
            dir_text = "/".join(dir_labels)
        else:
            dir_text = self._dir8_label(dir8) if dir8 else "unknown"
        
        return f"{base} ({dir_text})"
    
        
    def _alias_for_lane(self, lane_obj) -> str:
        meta = lane_obj.get("metadata", {}) or {}
        base = (meta.get("parent_road_name") or meta.get("road_name") or self.road_id2name.get(lane_obj.get("road_id"), "Road"))
        dir8 = (lane_obj.get("direction") or meta.get("dir8") or "unknown")
        turns = meta.get("allowed_turns") or []
        turn_txt = "left-turn" if "left" in turns else ("right-turn" if "right" in turns else ("through" if "through" in turns else "lane"))
        # crude lateral hint if present
        pos = meta.get("lateral_position_label") or meta.get("lane_position_label") or ""
        pos_txt = f" – {pos}" if pos else ""
        return f"{base} ({self._dir8_label(dir8)}) {turn_txt}{pos_txt}"

    def _lane_choices_for(self, road_id: int, dirset: set, intended_std: str):
        allowed_pref = {"straight": {"through"}, "left_turn": {"left"}, "right_turn": {"right"}}.get(intended_std, {"through","left","right"})
        def lane_dir(l): return (l.get("direction") or l.get("metadata", {}).get("dir8") or "").lower()
        def lane_ok(l):
            al = (l.get("metadata", {}).get("allowed_turns") or [])
            return any(a in allowed_pref for a in al) if al else True
        candidates = [l for l in self.lanes_by_road.get(road_id, []) if (lane_dir(l) in dirset)]
        prefer = [l for l in candidates if lane_ok(l)]
        return prefer or candidates

    def _dir8_from_origin_direction(self, origin_direction: str) -> set:
        d = (origin_direction or "unknown").lower()
        if "north" in d or d=="n": return {"s","se","sw"}
        if "south" in d or d=="s": return {"n","ne","nw"}
        if "west"  in d or d=="w": return {"e","ne","se"}
        if "east"  in d or d=="e": return {"w","nw","sw"}
        if d in ("ne","northeast"): return {"sw"}
        if d in ("nw","northwest"): return {"se"}
        if d in ("se","southeast"): return {"nw"}
        if d in ("sw","southwest"): return {"ne"}
        return {"n","s","e","w"}  # fall back


    def _build_road_conversational_map(self):
        """Build map storing ALL conversational identifiers"""
        self.road_conversational_map = {
            "roads": {},  # road_id -> primary conversational identifier
            "all_identifiers": {},  # road_id -> all conversational identifiers array
            "reverse_lookup": {}  # identifier -> road_id (for all identifiers)
        }
        
        for road in self.roads_data.get("roads_metadata", []):
            road_id = road.get("road_id")
            conv_identifiers = road.get("conversational_identifiers", [])
            user_descriptions = road.get("user_likely_descriptions", [])
            
            if road_id is not None and conv_identifiers:
                # Store primary identifier (first one)
                self.road_conversational_map["roads"][road_id] = conv_identifiers[0]
                
                # Store ALL identifiers
                self.road_conversational_map["all_identifiers"][road_id] = conv_identifiers
                
                # Build reverse lookup for ALL identifiers AND user descriptions
                for identifier in conv_identifiers + user_descriptions:
                    self.road_conversational_map["reverse_lookup"][identifier.lower()] = road_id
                
                logger.info(f"✅ Road {road_id}: {len(conv_identifiers)} conversational IDs")
                for i, identifier in enumerate(conv_identifiers):
                    logger.info(f"   {i+1}. \"{identifier}\"")

    def _propose_origin_roads_question(self):
        origin_dir = self.user_path.get("origin_direction")
        dirset = self._dir8_from_user_origin(self.standardize_direction(origin_dir))
        
        logger.info(f"🧭 User said: '{origin_dir}'")
        logger.info(f"🧭 Standardized: '{self.standardize_direction(origin_dir)}'")
        logger.info(f"🧭 Looking for lane directions: {dirset}")
        
        # Debug: Show all available lanes
        for rid, lanes in self.lanes_by_road.items():
            for lane in lanes:
                lane_dir = (lane.get("direction") or lane.get("metadata", {}).get("dir8") or "").lower()
                logger.info(f"🛣️ Road {rid} has lane with direction: '{lane_dir}'")
                
                # CRITICAL FIX: Check if this lane serves the user's origin direction
                if lane_dir in dirset:
                    matching_lanes.append(lane)
                    
            if matching_lanes:
                # Use conversational names
                conversational_id = self._get_conversational_name(rid)
                candidates.append((rid, conversational_id))
                if conversational_id:
                    candidates.append((rid, conversational_id))
                    logger.info(f"✅ Road {rid} -> '{conversational_id}'")
                else:
                    # Fallback to technical name only if no conversational_id
                    fallback_name = f"Road_{rid}"
                    candidates.append((rid, fallback_name))
                    logger.warning(f"⚠️ No conversational_id for road {rid}, using '{fallback_name}'")

        # Fallback: show all roads with conversational_identifiers
        if not candidates:
            logger.warning(f"⚠️ No directional matches, showing all available roads")
            for rid in self.road_conversational_map["roads"]:
                candidates.append((rid, self.road_conversational_map["roads"][rid]))

        if not candidates:
            logger.error("❗ No roads available!")
            return None

        candidates = candidates[:6]
        self._pending_origin_road_choices = [
            {"idx": i+1, "road_id": rid, "alias": alias} 
            for i, (rid, alias) in enumerate(candidates)
        ]
        
        options = "\n".join([f"{c['idx']}. {c['alias']}" for c in self._pending_origin_road_choices])
        
        return {
            "statusCode": 200,
            "message": f"Which road were you on when approaching the intersection?\n\n{options}\n\nPlease answer with the number (1, 2, 3, etc.)",
            "conversation_state": "choose_origin_road"
        }


    def _propose_origin_lanes_question(self):
        rid = self.user_path.get("origin_road_id")
        if rid is None: return None
        dirset = self._dir8_from_origin_direction(self.standardize_direction(self.user_path.get("origin_direction")))
        intended = self.standardize_maneuver(self.user_path.get("intended_maneuver"))
        lanes = self._lane_choices_for(rid, dirset, intended)
        if not lanes: return None
        self._pending_origin_lane_choices = [{"idx": i+1, "lane_id": l.get("lane_id"), "alias": self._alias_for_lane(l)} for i,l in enumerate(lanes[:6])]
        options = "\n".join([f"{c['idx']}. {c['alias']}" for c in self._pending_origin_lane_choices])
        return {
            "statusCode": 200,
            "message": f"Which lane were you in on that road?\n{options}\n\nPlease answer with a number.",
            "conversation_state": "choose_origin_lane"
        }


    def _confirmation_question(self):
        """Simple road-only confirmation"""
        road_display = self.user_path.get("origin_road_display", "selected road")
        maneu = self.standardize_maneuver(self.user_path.get("intended_maneuver"))
        
        maneu_txt = {
            "straight": "go straight",
            "left_turn": "turn left", 
            "right_turn": "turn right",
            "u_turn": "make a U-turn"
        }.get(maneu, maneu or "proceed")
        
        msg = (f"Just to confirm: you were on **{road_display}** "
            f"and planning to **{maneu_txt}**. Is that correct? (Yes/No)")
        
        return {
            "statusCode": 200, 
            "message": msg, 
            "conversation_state": "confirm_summary"
        }



    def _dir8_from_user_origin(self, origin_direction: str) -> set:
        """FIXED: Better direction mapping"""
        d = (origin_direction or "unknown").lower()
        
        # If user says "from north west", they're coming from NW, heading to intersection center
        # So we need lanes that serve traffic going FROM northwest TO the intersection
        if "north" in d and "west" in d:  # from northwest
            return {"nw", "northwest"}  # Look for NW-bound lanes (going toward center)
        elif "north" in d and "east" in d:  # from northeast  
            return {"ne", "northeast"}
        elif "south" in d and "west" in d:  # from southwest
            return {"sw", "southwest"}
        elif "south" in d and "east" in d:  # from southeast
            return {"se", "southeast"}
        elif "north" in d:
            return {"n", "north", "nb", "northbound"}
        elif "south" in d:
            return {"s", "south", "sb", "southbound"}
        elif "west" in d:
            return {"w", "west", "wb", "westbound"}
        elif "east" in d:
            return {"e", "east", "eb", "eastbound"}
        
        # Fallback
        return {"n", "s", "e", "w", "ne", "nw", "se", "sw"}

    def _maneuver_to_allowed(self, maneuver_std: str) -> set:
        if maneuver_std == "straight":    return {"through"}
        if maneuver_std == "left_turn":   return {"left"}
        if maneuver_std == "right_turn":  return {"right"}
        if maneuver_std == "u_turn":      return {"left"}  # most networks encode u-turn via left pocket
        return {"through","left","right"}

    def _get_lane_polyline(self, lane_obj):
        """Prefer map_coordinate_points; else fallback to points."""
        meta = lane_obj.get("metadata", {}) or {}
        pts  = meta.get("map_coordinate_points") or lane_obj.get("points") or []
        out = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
        return out

    def _resolve_vehicle_origin_ids(self, actor: str):
        """
        Resolve {road_id, lane_id} for 'user' or 'other'.
        Reads origin_* fields in self.user_path for that actor,
        plus intended maneuver to disambiguate lanes.
        """
        if actor == "user":
            origin_dir_raw   = self.user_path.get("origin_direction")
            origin_dir_std   = self.standardize_direction(origin_dir_raw)
            origin_road_name = (self.user_path.get("origin_road") or "").strip().lower()
            intended_std     = self.standardize_maneuver(self.user_path.get("intended_maneuver"))
        else:
            ov = self.user_path.get("other_vehicle", {}) or {}
            origin_dir_raw   = ov.get("origin_direction") or ov.get("position")  # best-effort
            origin_dir_std   = self.standardize_direction(origin_dir_raw)
            origin_road_name = (ov.get("origin_road") or "").strip().lower()
            intended_std     = self.standardize_maneuver(ov.get("action"))

        # (1) resolve road_id (prefer roads_metadata)
        rid = self.road_name2id.get(origin_road_name)
        if rid is None and origin_road_name:
            # fallback via lane metadata parent_road_name
            for (nm, _d), lanes in self.lanes_by_name_dir.items():
                if nm == origin_road_name and lanes:
                    rid = lanes[0].get("road_id")
                    break

        # Store road_id
        if actor == "user":
            self.user_path["origin_road_id"] = rid
        else:
            self.user_path.setdefault("other_vehicle", {})
            self.user_path["other_vehicle"]["origin_road_id"] = rid

        if rid is None:
            return  # cannot go further yet

        # (2) candidate lanes on that road
        candidates = list(self.lanes_by_road.get(rid, []))
        if not candidates:
            return

        # (3) filter by travel direction (toward intersection from the stated origin)
        dirset = self._dir8_from_origin_direction(origin_dir_std)
        def lane_dir(l):
            return (l.get("direction") or l.get("metadata", {}).get("dir8") or "").lower()
        c1 = [l for l in candidates if lane_dir(l) in dirset] or candidates

        # (4) filter by allowed turns vs intended maneuver (if known)
        allowed = self._maneuver_to_allowed(intended_std)
        def lane_allowed(l):
            al = (l.get("metadata", {}).get("allowed_turns") or [])
            return any(a in allowed for a in al) if al else True
        c2 = [l for l in c1 if lane_allowed(l)] or c1

        # (5) choose deterministically (first); store lane_id
        chosen = c2[0]
        if actor == "user":
            self.user_path["origin_lane_id"] = chosen.get("lane_id")
        else:
            self.user_path["other_vehicle"]["origin_lane_id"] = chosen.get("lane_id")

    # ---------- POLYLINE EMITTERS FOR BOTH VEHICLES ----------

    def _emit_route_points_from_lane(self, lane_id: str, label: str):
        lane = self.lane_by_id.get(lane_id)
        if not lane:
            return []
        poly = self._get_lane_polyline(lane)
        return [
            {
                "point_type": "lane_point",
                "description": label,
                "coordinates": {"x": float(x), "y": float(y)}
            } for (x, y) in poly
        ]


    def trigger_svg_generation(self, route_json):
        """Trigger enhanced SVG animation generation"""
        try:
            cluster_name = os.environ.get('CLUSTER_NAME')
            svg_task_def = os.environ.get('SVG_TASK_DEF')  # now expected to be "svg-animation-generator"
            subnet_id = os.environ.get('SUBNET_ID')
            security_group = os.environ.get('SECURITY_GROUP')

            if not all([cluster_name, svg_task_def, subnet_id, security_group]):
                logger.warning("⚠️ Missing ECS environment variables, skipping SVG generation")
                return

            # Family-based resolution logging
            if ":" not in str(svg_task_def):
                logger.info(f"🧭 Using family-only taskDefinition (LATEST will be resolved): {svg_task_def}")
            else:
                logger.warning(f"⚠️ SVG_TASK_DEF looks pinned to a revision/ARN: {svg_task_def}. "
                            f"Use family name (e.g., 'svg-animation-generator') to always pick LATEST.")

            response = ecs_client.run_task(
                cluster=cluster_name,
                taskDefinition=svg_task_def,  # now safe to pass family name
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': [subnet_id],
                        'securityGroups': [security_group],
                        'assignPublicIp': 'ENABLED'
                    }
                },
                overrides={
                    'containerOverrides': [
                        {
                            'name': 'svg-animation-generator',
                            'environment': [
                                {'name': 'CONNECTION_ID', 'value': self.connection_id},
                                {'name': 'BUCKET_NAME', 'value': self.bucket_name},
                                {'name': 'MODE', 'value': 'CLEAN_ANALYTICS'},
                                {'name': 'LLM_DATA_KEY', 'value': f"animation-data/{self.connection_id}/llm_ready_data.json"}
                            ]
                        }
                    ]
                }
            )

            task = response['tasks'][0]
            logger.info(f"🎬 run_task OK | taskArn={task['taskArn']} taskDefinitionArn={task.get('taskDefinitionArn')}")

        except Exception as e:
            logger.error(f"❗ Failed to trigger enhanced SVG generation: {e}")
            

def lambda_handler(event, context):
    """Main handler - supports both initial analysis and interactive conversation"""
    try:
        connection_id = event['connection_id']
        bucket_name = event['bucket_name']
        message_type = event.get('message_type', 'initial_analysis')
        
        logger.info(f"🔍 Analytics request: {message_type} for {connection_id}")
        
        if message_type == 'initial_analysis':
            return handle_initial_analysis(connection_id, bucket_name, event)
        elif message_type == 'user_response':
            return handle_user_response(connection_id, bucket_name, event)
        else:
            return {"statusCode": 400, "error": "Unknown message type"}
            
    except Exception as e:
        logger.error(f"❗ Analytics handler error: {e}")
        return {"statusCode": 500, "error": str(e)}


def handle_initial_analysis(connection_id, bucket_name, event):
    """Handle initial infrastructure analysis - FIXED to use normalized datetime"""
    try:
        logger.info(f"🚀 Starting initial analysis for {connection_id}")
        
        # Get collected data from the event
        collected_data = event.get('collected_data', {})
        logger.info(f"📥 Analytics received collected_data: {collected_data}")
        
        # ✅ NEW: Extract normalized datetime if available
        normalized_datetime = collected_data.get('normalized_datetime', 'unknown')
        if normalized_datetime and normalized_datetime != 'unknown':
            logger.info(f"✅ Using normalized datetime: {normalized_datetime}")
        
        # Load infrastructure data
        infrastructure_data = load_infrastructure_data(connection_id, bucket_name)
        logger.info(f"📁 Loaded {len(infrastructure_data)} infrastructure files")
        
        # Create analyzer
        analyzer = InteractiveTrafficAnalyzer(connection_id, bucket_name, infrastructure_data)
        
        # FIXED: Pass the collected data to the analyzer and pre-populate responses
        analyzer.existing_collected_data = collected_data
        analyzer.normalized_datetime = normalized_datetime  # ✅ NEW: Store normalized datetime
        
        # Pre-populate any info we already have with better key checking
        if normalized_datetime and normalized_datetime != 'unknown':
            # ✅ Use normalized datetime instead of raw input
            analyzer.responses['accident_time'] = normalized_datetime
            analyzer.user_path["accident_time"] = normalized_datetime
            logger.info(f"✅ Pre-populated normalized datetime: {normalized_datetime}")
        elif collected_data.get('basic_q_0'):  # Fallback to raw datetime
            analyzer.responses['accident_time'] = collected_data['basic_q_0']
            analyzer.user_path["accident_time"] = collected_data['basic_q_0']
            logger.info(f"⚠️ Using raw datetime: {collected_data['basic_q_0']}")
        
        if collected_data.get('basic_q_1'):  # description question  
            analyzer.responses['description'] = collected_data['basic_q_1']
            analyzer.user_path["accident_details"]["description"] = collected_data['basic_q_1']
            logger.info(f"✅ Pre-populated description: {collected_data['basic_q_1']}")
        
        # Do heavy analysis first
        logger.info("🔍 Starting infrastructure scene analysis...")
        scene_understanding = analyzer.analyze_infrastructure_scene()
        logger.info("✅ Infrastructure analysis complete")
        
        # Start interactive investigation
        logger.info("🕵️ Generating first investigation question...")
        first_question = analyzer.start_interactive_investigation()
        logger.info("✅ First question generated")
        
        # Save analyzer state for future interactions
        save_analyzer_state(connection_id, bucket_name, analyzer)
        logger.info("💾 Analyzer state saved")
        
        return {
            "statusCode": 200,
            "message": first_question,
            "conversation_state": "approach_direction",
            "scene_analysis": scene_understanding
        }
        
    except Exception as e:
        logger.error(f"❗ Initial analysis failed: {e}")
        return {"statusCode": 500, "error": str(e)}

def handle_user_response(connection_id, bucket_name, event):
    """Handle user response in interactive conversation - FIXED to preserve normalized_datetime"""
    try:
        user_input = event['user_input']
        current_state = event['conversation_state']
        normalized_datetime = event.get('normalized_datetime')  # ✅ NEW
        
        logger.info(f"👤 Processing user response: '{user_input}' in state: {current_state}")
        if normalized_datetime:
            logger.info(f"📅 Normalized datetime: {normalized_datetime}")
        
        # Load analyzer state
        analyzer = load_analyzer_state(connection_id, bucket_name)
        
        # ✅ NEW: Update normalized_datetime if provided
        if normalized_datetime and normalized_datetime != 'unknown':
            analyzer.normalized_datetime = normalized_datetime
        
        # Process user response
        result = analyzer.process_user_response(user_input, current_state)
        
        # Save updated state
        save_analyzer_state(connection_id, bucket_name, analyzer)
        
        logger.info(f"📊 Returning result: {result.get('conversation_state', 'unknown')}")
        
        return {
            "statusCode": 200,
            "message": result["message"],
            "conversation_state": result["conversation_state"],
            "animation_ready": result.get("animation_ready", False),
            "route_data": result.get("route_data"),
            "final_analysis": result.get("final_analysis")
        }
        
    except Exception as e:
        logger.error(f"❗ User response handling failed: {e}")
        return {"statusCode": 500, "error": str(e)}


def load_infrastructure_data(connection_id, bucket_name):
    """Load all infrastructure JSON files"""
    expected_files = [
        f'{connection_id}_roads_metadata_only.json',
        f'{connection_id}_metadata_only_network.json',
        f'{connection_id}_intersections_metadata_only.json',
        f'{connection_id}_lane_tree_routes_enhanced.json',  # NEW
    ]
    
    infrastructure_data = {}
    s3_prefix = f"outputs/{connection_id}/"
    
    for filename in expected_files:
        try:
            key = f"{s3_prefix}{filename}"
            response = s3.get_object(Bucket=bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            infrastructure_data[filename] = data
            logger.info(f"✅ Loaded {filename}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load {filename}: {e}")
    
    return infrastructure_data



def save_analyzer_state(connection_id, bucket_name, analyzer):
    """Save analyzer state for future interactions - FIXED to include normalized_datetime"""
    state_data = {
        "user_path": analyzer.user_path,
        "scene_understanding": analyzer.scene_understanding,
        "conversation_state": analyzer.investigation_steps[analyzer.current_step] if analyzer.current_step < len(analyzer.investigation_steps) else "complete",
        "current_step": analyzer.current_step,
        "responses": analyzer.responses,
        "normalized_datetime": getattr(analyzer, 'normalized_datetime', None),  # ✅ NEW
        "timestamp": datetime.now().isoformat()
    }
    
    state_key = f"analytics-state/{connection_id}/analyzer_state.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=state_key,
        Body=json.dumps(state_data, indent=2),
        ContentType='application/json'
    )


def load_analyzer_state(connection_id, bucket_name):
    """Load analyzer state from previous interactions - FIXED to restore normalized_datetime"""
    state_key = f"analytics-state/{connection_id}/analyzer_state.json"
    
    try:
        response = s3.get_object(Bucket=bucket_name, Key=state_key)
        state_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Load infrastructure data again
        infrastructure_data = load_infrastructure_data(connection_id, bucket_name)
        
        # Recreate analyzer with saved state
        analyzer = InteractiveTrafficAnalyzer(connection_id, bucket_name, infrastructure_data)
        analyzer.user_path = state_data["user_path"]
        analyzer.scene_understanding = state_data["scene_understanding"]
        analyzer.current_step = state_data["current_step"]
        analyzer.responses = state_data.get("responses", {})
        analyzer.normalized_datetime = state_data.get("normalized_datetime")  # ✅ NEW
        
        return analyzer
        
    except Exception as e:
        logger.error(f"❗ Failed to load analyzer state: {e}")
        raise