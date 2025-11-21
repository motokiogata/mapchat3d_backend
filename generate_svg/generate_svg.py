import os
import base64
import requests
from PIL import Image
from io import BytesIO
import json
import boto3
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_roads_only_style():
    """
    Return style parameters to show only roads without labels, symbols, or other elements
    """

    style_params = [
        # Hide all labels
        "&style=feature:all|element:labels|visibility:off",
        
        # Hide administrative boundaries
        "&style=feature:administrative|visibility:off",
        
        # Hide points of interest
        "&style=feature:poi|visibility:off",
        
        # Hide transit systems
        "&style=feature:transit|visibility:off",
        
        # Show only road geometry (no labels)
        "&style=feature:road|element:labels|visibility:off",
        "&style=feature:road|element:geometry|visibility:on",
        
        # Hide water labels but keep water bodies visible for context
        "&style=feature:water|element:labels|visibility:off",
        
        # Hide landscape labels
        "&style=feature:landscape|element:labels|visibility:off",
        
        # Specifically hide crosswalks and road markings
        "&style=feature:road|element:geometry.stroke|visibility:off",
        
        # Hide all road-related labels and symbols (including crosswalk symbols)
        "&style=feature:road|element:labels.icon|visibility:off",
        "&style=feature:road|element:labels.text|visibility:off",
        
        # Make roads more prominent with solid colors (no stripes/markings)
        "&style=feature:road.highway|element:geometry.fill|color:0x746855|visibility:on",
        "&style=feature:road.highway|element:geometry.stroke|visibility:off",
        "&style=feature:road.arterial|element:geometry.fill|color:0x746855|visibility:on", 
        "&style=feature:road.arterial|element:geometry.stroke|visibility:off",
        "&style=feature:road.local|element:geometry.fill|color:0x746855|visibility:on",
        "&style=feature:road.local|element:geometry.stroke|visibility:off",
    ]
    
    return "".join(style_params)

def get_google_map_image(lat, lng, zoom, size, scale, maptype, api_key, style_params=None):
    """
    Fetch Google Maps image with specified parameters
    """
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}"
        f"&zoom={zoom}&size={size}&scale={scale}&maptype={maptype}&key={api_key}"
    )
    
    # Add style parameters if provided
    if style_params:
        url += style_params
    
    logger.info(f"Fetching {maptype} image (zoom={zoom}, size={size})")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    return Image.open(BytesIO(response.content)).convert("RGBA")

def create_broad_map_and_svg(lat, lng, conn_id, api_key, s3_client, bucket_name):
    """
    Function 1: Create broad size map/satellite, generate SVG, and store as display versions
    Returns SVG for frontend
    """
    logger.info("=== FUNCTION 1: Creating broad map and SVG ===")
    
    # Broad map parameters (for display)
    zoom = 16  # Lower zoom for broader view
    size = "1280x1280"
    scale = 2
    
    # Fetch broad images
    sat_broad = get_google_map_image(lat, lng, zoom, size, scale, "satellite", api_key)
    road_broad = get_google_map_image(lat, lng, zoom, size, scale, "roadmap", api_key)
    
    # Create composite for SVG
    composite = Image.blend(sat_broad, road_broad, alpha=0.5)
    
    # Convert to base64 for SVG
    buffered = BytesIO()
    composite.save(buffered, format="PNG")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Create SVG
    img_width, img_height = composite.size
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{img_width}' height='{img_height}'>
    <image href='data:image/png;base64,{b64_data}' width='{img_width}' height='{img_height}' opacity='1.0'/>
    </svg>"""
    
    # Save locally (display versions - broad)
    sat_display_path = f"/tmp/{conn_id}_satellite_display.png"
    road_display_path = f"/tmp/{conn_id}_roadmap_display.png"
    
    sat_broad.save(sat_display_path, format="PNG")
    road_broad.save(road_display_path, format="PNG")
    
    # Upload display versions for frontend (broad ones)
    s3_client.upload_file(sat_display_path, bucket_name, f"{conn_id}_satellite_display.png")
    s3_client.upload_file(road_display_path, bucket_name, f"{conn_id}_roadmap_display.png")
    
    logger.info("✅ Broad maps (display versions) saved locally and uploaded to S3")
    
    return svg

def create_narrow_processing_maps(lat, lng, conn_id, api_key, s3_client, bucket_name):
    """
    Function 2: Create narrow size map/satellite optimized for processing 
    (zoom 20, size 640x640) and store as original versions
    """
    logger.info("=== FUNCTION 2: Creating narrow processing maps ===")
    
    # Narrow map parameters - optimized for processing
    zoom = 20  # High zoom for minimal fonts and detailed view
    size = "400x400"  # Optimal size for processing
    scale = 2
    
    # Get roads-only style parameters
    roads_style = get_roads_only_style()
    
    # Fetch narrow images
    sat_narrow = get_google_map_image(lat, lng, zoom, size, scale, "satellite", api_key)
    road_narrow = get_google_map_image(lat, lng, zoom, size, scale, "roadmap", api_key, roads_style)
    
    # Save locally (original versions - narrow for processing)
    sat_path = f"/tmp/{conn_id}_satellite.png"
    road_path = f"/tmp/{conn_id}_roadmap.png"
    
    sat_narrow.save(sat_path, format="PNG")
    road_narrow.save(road_path, format="PNG")
    
    # Upload analysis versions to S3 (original names - for processing)
    s3_client.upload_file(sat_path, bucket_name, f"{conn_id}_satellite.png")
    s3_client.upload_file(road_path, bucket_name, f"{conn_id}_roadmap.png")
    
    logger.info("✅ Narrow maps (original versions) saved locally and uploaded to S3")

def lambda_handler(event, context):
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "POST,OPTIONS"
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": ""
        }

    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        body = json.loads(event["body"])
        lat = body["latitude"]
        lng = body["longitude"]
        conn_id = body["connection_id"]
        
        logger.info(f"Processing request for lat={lat}, lng={lng}, conn_id={conn_id}")
        
        # Get configuration
        api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        if not api_key:
            logger.error("Google Maps API key not configured")
            raise ValueError("Google Maps API key not configured")
        
        bucket_name = os.environ["BUCKET_NAME"]
        s3_client = boto3.client("s3")
        
        # Execute both functions
        logger.info("🚀 Starting map generation process...")
        
        # Function 1: Create broad map and SVG (returns SVG for frontend)
        # Saves as: {conn_id}_satellite_display.png, {conn_id}_roadmap_display.png
        svg = create_broad_map_and_svg(lat, lng, conn_id, api_key, s3_client, bucket_name)
        
        # Function 2: Create narrow maps for processing (zoom 20, 640x640)
        # Saves as: {conn_id}_satellite.png, {conn_id}_roadmap.png (roadmap with roads-only styling)
        create_narrow_processing_maps(lat, lng, conn_id, api_key, s3_client, bucket_name)
        
        logger.info("✅ All map generation functions completed successfully")
        
        # Invoke next Lambda to start processing
        logger.info("Invoking process roadmap function...")
        lambda_client = boto3.client("lambda")
        lambda_client.invoke(
            FunctionName=os.environ['PROCESS_ROADMAP_FUNCTION'],
            InvocationType='Event',
            Payload=json.dumps({"base_name": conn_id})
        )
        logger.info("Successfully invoked process roadmap function")

        return {
            "statusCode": 200,
            "headers": {
                **cors_headers,
                "Content-Type": "image/svg+xml"
            },
            "body": svg
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": str(e)})
        }