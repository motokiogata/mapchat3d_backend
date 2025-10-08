import json
import boto3
import os

def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST,OPTIONS"
    }

    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": headers,
            "body": ""
        }

    try:
        body = json.loads(event.get("body", "{}"))
        connection_id = body.get("connection_id")
        
        # Debug print for connection_id
        print(f"DEBUG: Received connection_id: '{connection_id}'")
        
        if not connection_id:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({ "error": "Missing connection_id" })
            }

        bucket = os.environ["BUCKET_NAME"]
        s3 = boto3.client("s3")

        key_to_check = f"outputs/{connection_id}/done.flag"
        
        # Debug print for the S3 key being checked
        print(f"DEBUG: Checking S3 key: '{key_to_check}'")
        print(f"DEBUG: Bucket: '{bucket}'")
        
        try:
            s3.head_object(Bucket=bucket, Key=key_to_check)
            print(f"DEBUG: S3 object found - generating pre-signed URLs")
            
            # Define all the image files to return
            image_files = {
                # Root directory files
                "satellite": f"{connection_id}_satellite.png",
                "roadmap": f"{connection_id}_roadmap.png",
                
                # Files under /outputs/{connection_id}/
                "debug_mask": f"outputs/{connection_id}/{connection_id}_debug_mask_after_obstacle_removal.png",
                "debug_skeleton": f"outputs/{connection_id}/{connection_id}_debug_skeleton.png",
                "integrated_network": f"outputs/{connection_id}/{connection_id}_integrated_network_visualization.png",
                "comprehensive_narrative": f"outputs/{connection_id}/comprehensive_narrative_lane_visualization.png"
            }
            
            # Generate pre-signed URLs for each image
            image_urls = {}
            for image_name, image_key in image_files.items():
                try:
                    # Check if the object exists before generating URL
                    s3.head_object(Bucket=bucket, Key=image_key)
                    
                    # Generate pre-signed URL (valid for 1 hour)
                    url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': image_key},
                        ExpiresIn=3600  # 1 hour
                    )
                    image_urls[image_name] = url
                    print(f"DEBUG: Generated URL for {image_name}: {image_key}")
                    
                except s3.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        print(f"DEBUG: Image not found: {image_key}")
                        image_urls[image_name] = None
                    else:
                        print(f"DEBUG: Error checking {image_key}: {e}")
                        image_urls[image_name] = None
            
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({ 
                    "status": "done",
                    "images": image_urls
                })
            }
            
        except s3.exceptions.ClientError as e:
            print(f"DEBUG: S3 ClientError: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
            if e.response["Error"]["Code"] == "404":
                print(f"DEBUG: Object not found - returning 'processing'")
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps({ "status": "processing" })
                }
            else:
                raise e

    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({ "error": str(e) })
        }