import json
import boto3
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Chat history memory (keeps history while Lambda is warm)
chat_histories = {}

# AWS clients
region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table("AccidentDataTable")  # Replace with your actual table name

# Optional: Reuse API Gateway client
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
        connection_id = event["requestContext"]["connectionId"]
        body = json.loads(event.get("body", "{}"))
        user_msg = body.get("message", "Hello")
        jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))

        # Instruction prompt for Claude
        instruction = (
            f"Your name is Mariko, and you work for Tokio Marine Nichido, an insurance company, "
            f"as a kind and helpful operator handling traffic accident claims in English."
            f"You only handle cases involving car-to-car collisions that occur at intersections. You not able to accept reports for other types of accidents, such as those involving motorcycles, pedestrians, or incidents that occur on straight roads, highways, or in parking lots."
            f"Please ask the following two questions to gather information about the accident:\n\n"
            f"1. When did the accident occur? It is important to know the date and time of the accident. "
            f"FYI, today is \"{jst_time}\". "
            f"If the user just said an abstract time like 'today' or 'yesterday', you can guess it with the current date above."
            f"but please reconfirm the exact date to the user and make sure the guess is correct. "
            f"If the user said 'I don't know', confirm that they don't remember the time.\n\n"
            f"2. Where did the accident happen? Please gather detailed location information.\n\n"
            f"3. Then Please ask the customer to describe the details of the accident."
            f"Ask where their vehicle was coming from and where it was heading."
            f"Also ask which direction the other vehicle came from."
            f"If possible, gather information about the traffic signals and the speed of both vehicles."
            f"Let the customer know that it's fine to share only what they can remember.\n\n"
            f"After you have received answers to those questions above, say:\n"
            f"'I will now display the map in the area below. Please wait a moment.'\n"
            f"Then, end the conversation.\n\n"
            f"If the user asks anything outside of this task, politely decline to answer."
        )
        
        # Initialize chat history if new
        if connection_id not in chat_histories:
            chat_histories[connection_id] = [{"role": "user", "content": instruction}]

        # Append current user message
        chat_histories[connection_id].append({"role": "user", "content": user_msg})
        logger.info(f"📜 Chat history for {connection_id}: {chat_histories[connection_id]}")

        # Call Claude
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
        chat_histories[connection_id].append({"role": "assistant", "content": completion})

        # Step 1: Extract datetime and location (always run these)
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[connection_id]])
        datetime_str = extract_datetime(full_context)
        location_data = extract_location(full_context)
        
        # Step 2: Only run similarity search if LLM determines there are enough details
        similar_case = None
        if should_run_similarity_search(full_context):
            logger.info("🔍 Running similarity search - sufficient accident details detected")
            similar_case = find_similar_case(full_context)
            
            # Log similar case for debugging
            if similar_case:
                logger.info(f"🔍 Found similar cases: {len(similar_case)} cases")
                for i, case in enumerate(similar_case[:3]):  # Log top 3
                    logger.info(f"Case {i+1}: ID={case['case_id']}, Similarity={case['similarity']:.3f}")
            else:
                logger.info("🔍 No similar cases found")
        else:
            logger.info("⏭️ Skipping similarity search - insufficient accident details")

        # Step 3: Save to DynamoDB if valid
        if datetime_str != "unknown" and location_data:
            store_to_dynamodb(connection_id, datetime_str, location_data, similar_case)
            logger.info(f"✅ Stored to DynamoDB: {datetime_str}, {location_data}")
        else:
            logger.info("ℹ️ Datetime or location not available for storage.")

        # Step 4: Send Claude's reply back to frontend
        domain = event["requestContext"]["domainName"]
        stage = event["requestContext"]["stage"]
        apig = get_apig_client(domain, stage)

        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"response": completion}).encode("utf-8")
        )

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"❗ Unhandled error: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# ---------- Helper Functions ----------

def extract_datetime(full_context):
    prompt = f"Generate the exact datetime (format: yyyy/mm/dd hh:mm) from this accident description: \"{full_context}\". If the user provides a relative date expression such as yesterday, today, or three days ago, instead of a specific date, estimate the exact date by refferring today's date. Just output yyyy/mm/dd hh:mm directly and Don't include any other messages. If not you can't do it, return 'unknown'."
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    reply = json.loads(response["body"].read())["content"][0]["text"].strip()
    print("Datetime Response:", reply)
    return reply


import re

def extract_location(full_context):
    prompt = (
        f"Generate the location in JSON format like {{\"lat\": 35.6895, \"lon\": 139.6917}} "
        f"from this message: \"{full_context}\". "
        f"If you're not confident, still it's OK. Please assume as much as possible because the user can correct you. "
        f"So don't hesitate."
        f"Don't include triple backticks or any Markdown formatting. Just output the JSON object directly."
    )

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )

    raw_reply = response["body"].read()
    reply_text = json.loads(raw_reply)["content"][0]["text"].strip()
    print("Location Response:", repr(reply_text))

    # Remove Markdown-style formatting (```json ... ```)
    match = re.search(r'\{.*\}', reply_text, re.DOTALL)
    if match:
        cleaned = match.group(0)
        try:
            location = json.loads(cleaned)
            if isinstance(location, dict) and "lat" in location and "lon" in location:
                return location
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse cleaned location: {e}")
    else:
        logger.warning("⚠️ No JSON object found in response")

    return None


def store_to_dynamodb(connection_id, datetime_str, location, similar_cases=None):
    response = table.get_item(Key={"connection_id": connection_id})
    item = response.get("Item")

    # Prepare similar cases data for storage
    similar_cases_data = None
    if similar_cases and len(similar_cases) > 0:
        # Store top 3 similar cases
        similar_cases_data = []
        for case in similar_cases[:3]:
            similar_cases_data.append({
                "case_id": str(case["case_id"]),
                "similarity": Decimal(str(case["similarity"])),  # ✅ Convert float to Decimal
                "summary": case["summary"][:500]  # Limit length for DynamoDB
            })

    if item:
        # Update the existing item
        update_expression = "SET #ts = :ts, lat = :lat, lon = :lon"
        expression_values = {
            ":ts": datetime_str,
            ":lat": Decimal(str(location["lat"])),
            ":lon": Decimal(str(location["lon"]))
        }
        
        if similar_cases_data:
            update_expression += ", similar_cases = :similar"
            expression_values[":similar"] = similar_cases_data
            
        table.update_item(
            Key={"connection_id": connection_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames={"#ts": "timestamp"},
            ExpressionAttributeValues=expression_values
        )
    else:
        # Insert new item
        new_item = {
            "connection_id": connection_id,
            "timestamp": datetime_str,
            "lat": Decimal(str(location["lat"])),
            "lon": Decimal(str(location["lon"]))
        }
        
        if similar_cases_data:
            new_item["similar_cases"] = similar_cases_data
            
        table.put_item(Item=new_item)


def find_similar_case(full_context):
    """
    Call the similarity search Lambda function to find similar cases
    """
    try:
        # Get function name from environment
        function_name = os.environ.get('SIMILARITY_FUNCTION_NAME')
        if not function_name:
            logger.warning("SIMILARITY_FUNCTION_NAME not set")
            return None
            
        # Invoke similarity search Lambda
        lambda_client = boto3.client('lambda')
        
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'user_text': full_context,
                'top_k': 5  # Get top 5 similar cases
            })
        )
        
        result = json.loads(response['Payload'].read())
        if result['statusCode'] == 200:
            response_data = json.loads(result['body'])
            similar_cases = response_data['similar_cases']
            logger.info(f"✅ Found {len(similar_cases)} similar cases from {response_data['total_cases_checked']} total cases")
            
            # Log similarity scores for debugging
            for i, case in enumerate(similar_cases[:3]):
                logger.info(f"Top case {i+1}: Case {case['case_id']} - Similarity: {case['similarity']:.4f}")
            
            return similar_cases
        else:
            logger.error(f"Similarity search failed: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling similarity search: {e}")
        return None

#an LLM gate to determine when to run the similarity search

def should_run_similarity_search(full_context):
    """
    Use LLM to determine if the conversation contains enough accident details 
    to warrant running similarity search
    """
    try:
        prompt = (
            f"Analyze this conversation and determine if the user has provided enough details "
            f"about their traffic accident to run a similarity search against a database of accident cases.\n\n"
            f"The user should have provided:\n"
            f"1. Basic accident details (what happened, intersection collision)\n"
            f"2. Vehicle movements (which direction each car was going, turning, straight, etc.)\n"
            f"3. Traffic conditions (signals, signs, right of way, etc.)\n\n"
            f"Conversation:\n{full_context}\n\n"
            f"Respond with only 'YES' if there are enough accident details for similarity search, "
            f"or 'NO' if more details are needed. Don't include any other text."
        )
        
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.1,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        
        reply = json.loads(response["body"].read())["content"][0]["text"].strip().upper()
        logger.info(f"🤖 Similarity search decision: {reply}")
        
        return reply == "YES"
        
    except Exception as e:
        logger.error(f"Error in similarity search decision: {e}")
        # Default to not running similarity search if there's an error
        return False


# ---------- Entry Point ----------

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