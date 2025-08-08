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
        
        zoom = 18
        size = "1280x1280"
        scale = 2

        key = os.environ.get('GOOGLE_MAPS_API_KEY')
        if not key:
            logger.error("Google Maps API key not configured")
            raise ValueError("Google Maps API key not configured")

        def fetch_img(maptype):
            url = (
                f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}"
                f"&zoom={zoom}&size={size}&scale={scale}&maptype={maptype}&key={key}"
            )
            logger.info(f"Fetching {maptype} image from: {url[:100]}...")  # Log partial URL for security
            r = requests.get(url, timeout=30)  # Add timeout
            r.raise_for_status()
            logger.info(f"Successfully fetched {maptype} image")
            return Image.open(BytesIO(r.content)).convert("RGBA")

        logger.info("Fetching satellite image...")
        sat = fetch_img("satellite")
        
        logger.info("Fetching roadmap image...")
        road = fetch_img("roadmap")

        logger.info("Creating composite image...")
        # Blend for return preview SVG
        composite = Image.blend(sat, road, alpha=0.5)
        buffered = BytesIO()
        composite.save(buffered, format="PNG")
        b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # After creating composite
        img_width, img_height = composite.size
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{img_width}' height='{img_height}'>
        <image href='data:image/png;base64,{b64_data}' width='{img_width}' height='{img_height}' opacity='1.0'/>
        </svg>"""

        logger.info("Uploading images to S3...")
        # Save to /tmp and upload to S3
        s3 = boto3.client("s3")
        bucket_name = os.environ["BUCKET_NAME"]
        base_name = conn_id

        sat_path = f"/tmp/{base_name}_satellite.png"
        road_path = f"/tmp/{base_name}_roadmap.png"

        sat.save(sat_path, format="PNG")
        road.save(road_path, format="PNG")

        s3.upload_file(sat_path, bucket_name, f"{base_name}_satellite.png")
        s3.upload_file(road_path, bucket_name, f"{base_name}_roadmap.png")
        logger.info("Successfully uploaded images to S3")

        logger.info("Invoking process roadmap function...")
        # Optionally invoke next Lambda to start Fargate
        lambda_client = boto3.client("lambda")
        lambda_client.invoke(
            FunctionName=os.environ['PROCESS_ROADMAP_FUNCTION'],
            InvocationType='Event',
            Payload=json.dumps({"base_name": base_name})
        )
        logger.info("Successfully invoked process roadmap function")

        logger.info("Returning SVG response")
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