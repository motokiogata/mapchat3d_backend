# bedrock_analyzer.py
import boto3
import json
import base64
from config import BEDROCK_MODEL_ID, BEDROCK_REGION

class BedrockAnalyzer:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for Bedrock API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_roadmap_with_bedrock(self, roadmap_path, satellite_path):
        """Analyze images with AWS Bedrock Claude for user-friendly aliases"""
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
            
            response = self.bedrock.invoke_model(
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