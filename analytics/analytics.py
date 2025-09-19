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
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
table = dynamodb.Table("AccidentDataTable")

class InteractiveTrafficAnalyzer:
    def __init__(self, connection_id, bucket_name, infrastructure_data):
        self.connection_id = connection_id
        self.bucket_name = bucket_name
        self.infrastructure = infrastructure_data
        
        # ✅ KEEP JSON DATA ACCESSIBLE THROUGHOUT
        self.roads_data = infrastructure_data.get(f'{connection_id}_roads_metadata_only.json', {})
        self.network_data = infrastructure_data.get(f'{connection_id}_metadata_only_network.json', {})
        self.intersections_data = infrastructure_data.get(f'{connection_id}_intersections_metadata_only.json', {})
        
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
        """Generate direction question using intersections and roads JSON data"""
        prompt = f"""
        Based on this intersection and road data, generate a simple question asking which direction the user came from:
        
        INTERSECTIONS DATA:
        {json.dumps(self.intersections_data, indent=2)[:3000]}
        
        ROADS DATA:
        {json.dumps(self.roads_data, indent=2)[:3000]}
        
        SCENE UNDERSTANDING:
        {self.scene_understanding}
        
        Generate a question like:
        "I can see this accident occurred at [specific intersection description from the data]. Could you tell me which direction you were coming from when you approached this intersection?"
        
        Use the actual road names and intersection details from the JSON data.
        Keep it simple and clear - ONE question only.
        """
        
        question = self.call_bedrock(prompt, max_tokens=300)
        
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
        """Generate other vehicle question using all JSON data"""
        prompt = f"""
        Based on the intersection layout and user's path so far, ask about the other vehicle:
        
        INTERSECTION DATA:
        {json.dumps(self.intersections_data, indent=2)[:2000]}
        
        USER'S PATH SO FAR:
        {json.dumps(self.responses, indent=2)}
        
        Ask where the other vehicle was when they first noticed it.
        Reference the intersection layout from the JSON if helpful.
        
        Keep it simple: "Where was the other vehicle when you first noticed it?"
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
        """Generate time question - skip if already have datetime info"""
        
        # Check if we already have datetime from earlier collection
        if hasattr(self, 'existing_collected_data') and self.existing_collected_data.get('basic_q_0'):
            # We already have this info, skip this question
            logger.info("⏭️ Skipping time question - already have datetime info")
            self.responses['accident_time'] = self.existing_collected_data['basic_q_0']
            self.user_path["accident_time"] = self.existing_collected_data['basic_q_0']
            
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
        """Process response and move to next question"""
        
        # Save response
        self.responses[current_state] = user_input
        
        # Update user path with structured data
        self.update_user_path_from_response(user_input, current_state)
        
        # Move to next step
        self.current_step += 1
        
        # Get next question (which will use JSON data)
        return self.get_next_question()

    def update_user_path_from_response(self, user_input, current_state):
        """Update user path with structured data extraction"""
        
        if current_state == "approach_direction":
            self.user_path["origin_direction"] = user_input
            # Extract road name if possible
            self.extract_road_name_from_direction(user_input)
            
        elif current_state == "intended_action":
            self.user_path["intended_maneuver"] = user_input
            # Extract destination road if turning
            self.extract_destination_from_action(user_input)
            
        elif current_state == "lane_position":
            self.user_path["approach_lane"] = user_input
            
        elif current_state == "traffic_signal":
            self.user_path["traffic_conditions"]["signal_state"] = user_input
            
        elif current_state == "other_vehicle_position":
            self.user_path["other_vehicle"]["position"] = user_input
            
        elif current_state == "other_vehicle_action":
            self.user_path["other_vehicle"]["action"] = user_input
            
        elif current_state == "collision_point":
            self.user_path["collision_point"] = user_input
            
        elif current_state == "accident_time":
            self.user_path["accident_time"] = user_input

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
            self.user_path["origin_road"] = road_name.strip()
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
        """Generate final summary and create route JSON"""
        logger.info("🏁 Generating completion summary and route JSON")
        
        # Generate analysis summary
        summary_prompt = f"""
        Generate a professional accident reconstruction summary based on these responses:
        
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
        
        Write as a professional accident reconstruction report.
        """
        
        final_summary = self.call_bedrock(summary_prompt, max_tokens=1000)
        
        # ✅ CREATE ROUTE JSON
        route_json = self.create_route_json()
        
        # ✅ SAVE TO DYNAMODB
        self.save_to_dynamodb(route_json, final_summary)
        
        # Save animation data
        self.save_animation_data(route_json)

        # 🎬 NEW: Trigger SVG animation generation
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
                            "road_name": self.user_path.get("origin_road", "unknown"),
                            "lane": self.standardize_lane(self.user_path.get("approach_lane"))
                        },
                        "intended_destination": {
                            "maneuver": self.standardize_maneuver(self.user_path.get("intended_maneuver")),
                            "target_road": self.user_path.get("destination_road", "unknown")
                        },
                        "route_points": self.generate_user_route_points()
                    },
                    "state_at_collision": {
                        "speed": "unknown",  # Could be estimated later
                        "position": self.user_path.get("collision_point", "unknown"),
                        "action": self.user_path.get("intended_maneuver", "unknown")
                    }
                },
                
                "other_vehicle": {
                    "path": {
                        "origin": {
                            "direction": self.estimate_other_vehicle_origin(),
                            "road_name": "unknown",
                            "lane": "unknown"
                        },
                        "intended_destination": {
                            "maneuver": self.standardize_maneuver(self.user_path.get("other_vehicle", {}).get("action")),
                            "target_road": "unknown"
                        },
                        "route_points": self.generate_other_vehicle_route_points()
                    },
                    "state_at_collision": {
                        "speed": "unknown",
                        "position": self.user_path.get("other_vehicle", {}).get("position", "unknown"),
                        "action": self.user_path.get("other_vehicle", {}).get("action", "unknown")
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
                    "route_json": route_json,
                    "final_summary": final_summary,
                    "investigation_complete": True,
                    "completed_at": datetime.now().isoformat()
                }
            }
            
            # Merge with existing data
            item.update(update_data)
            
            # Convert to DynamoDB format
            item_dynamodb = self.convert_to_dynamodb_format(item)
            
            # Save to DynamoDB
            table.put_item(Item=item_dynamodb)
            
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

    def save_animation_data(self, route_json):
        """Save data needed for animation creation"""
        animation_data = {
            "user_path": self.user_path,
            "infrastructure": self.scene_understanding,
            "route_json": route_json,
            "timeline": self.user_path.get("timeline", []),
            "accident_point": self.user_path.get("collision_point"),
            "created_at": datetime.now().isoformat()
        }
        
        # Save to S3 for future animation creation
        animation_key = f"animation-data/{self.connection_id}/complete_route_data.json"
        s3.put_object(
            Bucket=self.bucket_name,
            Key=animation_key,
            Body=json.dumps(animation_data, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"💾 Animation data saved: {animation_key}")
    
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

    def trigger_svg_generation(self, route_json):
        """Trigger SVG animation generation using ECS Fargate task"""
        try:
            # Use ECS client to run Fargate task instead of Lambda invoke
            ecs_client = boto3.client('ecs')
            
            response = ecs_client.run_task(
                cluster=os.environ['CLUSTER_NAME'],
                taskDefinition=os.environ['SVG_TASK_DEF'],
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': [os.environ['SUBNET_ID']],
                        'securityGroups': [os.environ['SECURITY_GROUP']],
                        'assignPublicIp': 'ENABLED'
                    }
                },
                overrides={
                    'containerOverrides': [
                        {
                            'name': 'svg-animation-generator',
                            'environment': [
                                {
                                    'name': 'CONNECTION_ID',
                                    'value': self.connection_id
                                },
                                {
                                    'name': 'BUCKET_NAME',
                                    'value': self.bucket_name
                                }
                            ]
                        }
                    ]
                }
            )
            
            task_arn = response['tasks'][0]['taskArn']
            logger.info(f"🎬 SVG animation Fargate task started: {task_arn}")
            
        except Exception as e:
            logger.error(f"❗ Failed to trigger SVG generation: {e}")



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
    """Handle initial infrastructure analysis"""
    try:
        logger.info(f"🚀 Starting initial analysis for {connection_id}")
        
        # Get collected data from the event
        collected_data = event.get('collected_data', {})
        
        # Load infrastructure data
        infrastructure_data = load_infrastructure_data(connection_id, bucket_name)
        logger.info(f"📁 Loaded {len(infrastructure_data)} infrastructure files")
        
        # Create analyzer
        analyzer = InteractiveTrafficAnalyzer(connection_id, bucket_name, infrastructure_data)
        
        # FIXED: Pass the collected data to the analyzer
        analyzer.existing_collected_data = collected_data
        
        # Pre-populate any time/date info we already have
        if collected_data.get('basic_q_0'):  # datetime question
            analyzer.responses['accident_time'] = collected_data['basic_q_0']
            analyzer.user_path["accident_time"] = collected_data['basic_q_0']
        
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
        
        # Return directly - no S3 saving needed
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
    """Handle user response in interactive conversation"""
    try:
        user_input = event['user_input']
        current_state = event['conversation_state']
        
        logger.info(f"👤 Processing user response: '{user_input}' in state: {current_state}")
        
        # Load analyzer state
        analyzer = load_analyzer_state(connection_id, bucket_name)
        
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
        f'{connection_id}_intersections_metadata_only.json'
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
    """Save analyzer state for future interactions"""
    state_data = {
        "user_path": analyzer.user_path,
        "scene_understanding": analyzer.scene_understanding,
        "conversation_state": analyzer.investigation_steps[analyzer.current_step] if analyzer.current_step < len(analyzer.investigation_steps) else "complete",
        "current_step": analyzer.current_step,
        "responses": analyzer.responses,
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
    """Load analyzer state from previous interactions"""
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
        
        return analyzer
        
    except Exception as e:
        logger.error(f"❗ Failed to load analyzer state: {e}")
        raise