# config.py
import os

# AWS Bedrock setup
BEDROCK_MODEL_ID = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
BEDROCK_REGION = "ap-northeast-1"

# File paths
MASK_PATH = "final_road_mask_cleaned.png"
ROADMAP_PATH = "roadmap.png"
SATELLITE_PATH = "satellite.png"

# Output files
OUTPUT_INTEGRATED_JSON = "integrated_road_network.json"
OUTPUT_CENTERLINES_JSON = "centerlines_with_metadata.json"
OUTPUT_INTERSECTIONS_JSON = "intersections_with_metadata.json"
OUTPUT_LANE_TREES_JSON = "lane_tree_routes_enhanced.json"
OUTPUT_IMG = "integrated_network_visualization.png"
DEBUG_SKELETON = "debug_skeleton.png"

# Processing parameters
MIN_LINE_LENGTH = 20
CANVAS_SIZE = (1280, 1280)
EDGE_TOLERANCE = 10
LANE_OFFSET_PX = 20
INTERSECTION_RADIUS = 30

# S3 Configuration
S3_BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-output-bucket")