import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Memory for storing chat history while Lambda is warm
chat_histories = {}

region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)

# Optional: reuse client across invocations
apig_clients = {}

def get_apig_client(domain, stage):
    endpoint = f"https://{domain}/{stage}"
    if endpoint not in apig_clients:
        apig_clients[endpoint] = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)
    return apig_clients[endpoint]

def handle_connect(event):
    connection_id = event["requestContext"]["connectionId"]
    logger.info(f"✅ Client connected: {connection_id}")
    return {"statusCode": 200}

def handle_disconnect(event):
    connection_id = event["requestContext"]["connectionId"]
    logger.info(f"❌ Client disconnected: {connection_id}")
    return {"statusCode": 200}

def handle_send_message(event):
    try:
        connection_id = event['requestContext']['connectionId']
        body = json.loads(event.get('body', '{}'))
        user_msg = body.get('message', 'Hello')

        instruction = (
            "Your name is Mariko and you work for Tokio Marine Nichido, "
            "the insurance company, as a helpful operator handling claims "
            "for traffic accidents. Ask the user one question at a time "
            "to gather the following information step by step: time, location, "
            "vehicles involved, damages, injuries, and insurance information. "
            "Only move to the next topic after the current one is answered. "
            "If the user asks something outside of this task, politely decline to answer."
        )

        # Initialize chat history for this connection if it doesn't exist
        if connection_id not in chat_histories:
            # Claude expects the instruction as the first user message
            chat_histories[connection_id] = [
                { "role": "user", "content": instruction }
            ]

        # Add the actual user message next
        chat_histories[connection_id].append({ "role": "user", "content": user_msg })
        print(f"Chat history for {connection_id}: {chat_histories[connection_id]}")

        # Call Claude with full message history
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.1,
                "messages": chat_histories[connection_id]
            })
        )

        response_body = json.loads(response["body"].read())
        completion = response_body["content"][0]["text"]

        # Add Claude's reply to the chat history
        chat_histories[connection_id].append({ "role": "assistant", "content": completion })

        # Send back to frontend via WebSocket
        domain = event["requestContext"]["domainName"]
        stage = event["requestContext"]["stage"]
        apig = get_apig_client(domain, stage)

        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({ "response": completion }).encode("utf-8")
        )

        return { "statusCode": 200 }

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return { "statusCode": 500, "body": "Internal Server Error" }

# Main Lambda handler
# This function routes the incoming WebSocket events to the appropriate handler
def lambda_handler(event, context):
    route_key = event.get("requestContext", {}).get("routeKey")

    if route_key == "$connect":
        return handle_connect(event)
    elif route_key == "$disconnect":
        return handle_disconnect(event)
    elif route_key == "sendMessage":
        return handle_send_message(event)
    else:
        logger.error(f"Unknown route: {route_key}")
        return {"statusCode": 400, "body": "Unsupported route"}
