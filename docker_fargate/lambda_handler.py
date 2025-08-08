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
        if not connection_id:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({ "error": "Missing connection_id" })
            }

        bucket = os.environ["BUCKET_NAME"]
        s3 = boto3.client("s3")

        key_to_check = f"outputs/{connection_id}/integrated_road_network.json"
        try:
            s3.head_object(Bucket=bucket, Key=key_to_check)
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({ "status": "done" })
            }
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps({ "status": "processing" })
                }
            else:
                raise e

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({ "error": str(e) })
        }
