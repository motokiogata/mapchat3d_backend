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
        
        print(f"DEBUG: Received connection_id: '{connection_id}'")
        
        if not connection_id:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({ "error": "Missing connection_id" })
            }

        # DynamoDB client
        dynamodb = boto3.client('dynamodb')
        table_name = 'AccidentDataTable'
        
        print(f"DEBUG: Checking DynamoDB for connection_id: '{connection_id}'")
        
        try:
            response = dynamodb.get_item(
                TableName=table_name,
                Key={
                    'connection_id': {'S': connection_id}
                }
            )
            
            if 'Item' in response and 'route_analysis' in response['Item']:
                route_analysis = response['Item']['route_analysis']['M']
                
                # Check if investigation is complete
                if route_analysis.get('investigation_complete', {}).get('BOOL', False):
                    print(f"DEBUG: Report found and complete")
                    
                    # Extract the data
                    report_data = {
                        "status": "ready",
                        "completed_at": route_analysis.get('completed_at', {}).get('S', ''),
                        "final_summary": route_analysis.get('final_summary', {}).get('S', ''),
                        "route_s3_key": route_analysis.get('route_s3_key', {}).get('S', '')
                    }
                    
                    return {
                        "statusCode": 200,
                        "headers": headers,
                        "body": json.dumps(report_data)
                    }
                else:
                    print(f"DEBUG: Report found but investigation not complete")
                    return {
                        "statusCode": 200,
                        "headers": headers,
                        "body": json.dumps({ "status": "processing" })
                    }
            else:
                print(f"DEBUG: No report data found")
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps({ "status": "processing" })
                }
                
        except Exception as e:
            print(f"DEBUG: DynamoDB error: {str(e)}")
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({ "status": "processing" })
            }

    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({ "error": str(e) })
        }