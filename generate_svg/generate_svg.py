import os
import base64
import requests
from PIL import Image
from io import BytesIO
import json

def lambda_handler(event, context):
    # CORS headers for REST API
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "POST,OPTIONS"
    }
    
    # ✅ Handle preflight request (REST API structure)
    if event.get('httpMethod') == 'OPTIONS':
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": ""
        }
    
    try:
        body = json.loads(event["body"])
        lat = body["latitude"]
        lng = body["longitude"]
        zoom = 20
        size = "1280x1280"
        scale = 2
        
        # ✅ Get API key from environment variable
        key = os.environ.get('GOOGLE_MAPS_API_KEY')
        if not key:
            raise ValueError("Google Maps API key not configured")

        def fetch_img(maptype):
            url = (
                f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}"
                f"&zoom={zoom}&size={size}&scale={scale}&maptype={maptype}&key={key}"
            )
            r = requests.get(url)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGBA")

        sat = fetch_img("satellite")
        road = fetch_img("roadmap")

        composite = Image.blend(sat, road, alpha=0.5)

        buffered = BytesIO()
        composite.save(buffered, format="PNG")
        b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='500' height='500'>
          <image href='data:image/png;base64,{b64_data}' width='500' height='500' opacity='1.0'/>
        </svg>"""

        return {
            "statusCode": 200,
            "headers": {
                **cors_headers,
                "Content-Type": "image/svg+xml"
            },
            "body": svg
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": str(e)})
        }