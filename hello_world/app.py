import json
import boto3
import os
import logging
import re
import pandas as pd
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

# --- Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table("AccidentDataTable")
s3 = boto3.client("s3")
CSV_BUCKET = os.environ.get('CSV_BUCKET_NAME', 'your-csv-bucket')

chat_histories = {}
apig_clients = {}

CATEGORY_CSV_MAPPING = {
    "Vehicle(car_or_motorcycle)_accident_against_pedestrian": ["1-25.csv", "26-50.csv"],
    "Bycicle_accident_against_pedestrian": ["51-74.csv", "75-97.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_motorcycle)": ["98-113.csv", "114-138.csv", "139-159.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_car)": ["160-204.csv", "205-234.csv"],
    "Vehicle(car_or_motorcycle)_accident_against_bycicle": ["235-280.csv", "281-310.csv"],
    "Accidents_in_highways_or_Accidents_in_park": ["311-338.csv"]
}

# --- Connection handlers ---
def get_apig_client(domain, stage):
    endpoint = f"https://{domain}/{stage}"
    if endpoint not in apig_clients:
        apig_clients[endpoint] = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)
    return apig_clients[endpoint]

def handle_connect(event):
    logger.info(f"✅ Client connected: {event['requestContext']['connectionId']}")
    return {"statusCode": 200}

def handle_disconnect(event):
    logger.info(f"❌ Client disconnected: {event['requestContext']['connectionId']}")
    return {"statusCode": 200}

# --- Main Message Handler ---
def handle_send_message(event):
    try:
        connection_id = event["requestContext"]["connectionId"]
        body = json.loads(event.get("body", "{}"))
        user_msg = body.get("message", "Hello")
        jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))

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
            f"'I will now display the accident description below. Please wait a moment.'\n"
            f"Then, end the conversation.\n\n"
            f"If the user asks anything outside of this task, politely decline to answer."
        )

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

        completion = json.loads(response["body"].read())["content"][0]["text"]
        chat_histories[connection_id].append({"role": "assistant", "content": completion})

        # Step 1: Extract datetime and location (always run these)
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[connection_id]])
        datetime_str = extract_datetime(full_context)
        location_data = extract_location(full_context)

        similar_case = None
        if should_run_similarity_search(full_context):
            similar_case = find_similar_case(full_context)

        if datetime_str != "unknown" and location_data:
            store_to_dynamodb(connection_id, datetime_str, location_data, similar_case)

        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(ConnectionId=connection_id, Data=json.dumps({"response": completion}).encode("utf-8"))

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"❗ Unhandled error: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# --- Helpers ---
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


def should_run_similarity_search(full_context):
    prompt = f"Analyze this conversation and determine ..."
    response = bedrock.invoke_model(...)
    reply = json.loads(response["body"].read())["content"][0]["text"].strip().upper()
    return reply == "YES"

# --- CSV Similarity Logic ---
def categorize_accident_type(full_context):
    prompt = "..."
    response = bedrock.invoke_model(...)
    return json.loads(response["body"].read())["content"][0]["text"].strip()

def load_csv_from_s3(csv_filename):
    response = s3.get_object(Bucket=CSV_BUCKET, Key=csv_filename)
    return pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

def format_cases_for_analysis(cases):
    return "\n\n".join([f"Case {c['case_number']}: Situation: {c['situation']} ..." for c in cases[:20]])

def find_matching_pattern(category, full_context):
    all_cases = []
    for file in CATEGORY_CSV_MAPPING.get(category, []):
        df = load_csv_from_s3(file)
        for _, row in df.iterrows():
            all_cases.append({"case_number": row["Case Number"], ...})
    prompt = f"Based on this accident description, find the MOST SIMILAR case ..."
    response = bedrock.invoke_model(...)
    result_text = json.loads(response["body"].read())["content"][0]["text"]
    json_match = re.search(r'\{.*\}', result_text)
    return json.loads(json_match.group(0)) if json_match else None

def calculate_final_fault_ratio(base_ratio, modifications, accident_context):
    prompt = f"Calculate the final fault ratio ..."
    response = bedrock.invoke_model(...)
    return json.loads(response["body"].read())["content"][0]["text"].strip()

def find_similar_case(full_context):
    category = categorize_accident_type(full_context)
    match = find_matching_pattern(category, full_context)
    if match:
        final = calculate_final_fault_ratio(match["fault_ratio"], match.get("applicable_modifications", []), full_context)
        return [{
            "case_id": match["best_match_case_number"],
            "category": category,
            "similarity": float(match["confidence_score"]) / 10.0,
            "summary": match["reasoning"],
            "fault_ratio": final,
            "base_fault_ratio": match["fault_ratio"],
            "modifications": match.get("applicable_modifications", [])
        }]
    return None

def store_to_dynamodb(connection_id, datetime_str, location, similar_cases=None):
    item = table.get_item(Key={"connection_id": connection_id}).get("Item")
    similar_data = None
    if similar_cases:
        similar_data = [{
            "case_id": case["case_id"],
            "category": case.get("category", "unknown"),
            "similarity": Decimal(str(case["similarity"])),
            "summary": case["summary"][:500],
            "fault_ratio": case.get("fault_ratio", "unknown"),
            "base_fault_ratio": case.get("base_fault_ratio", "unknown"),
            "modifications": case.get("modifications", [])[:5]
        } for case in similar_cases[:3]]
    values = {
        ":ts": datetime_str,
        ":lat": Decimal(str(location["lat"])),
        ":lon": Decimal(str(location["lon"]))
    }
    if similar_data:
        values[":similar"] = similar_data

    if item:
        table.update_item(
            Key={"connection_id": connection_id},
            UpdateExpression="SET #ts = :ts, lat = :lat, lon = :lon, similar_cases = :similar",
            ExpressionAttributeNames={"#ts": "timestamp"},
            ExpressionAttributeValues=values
        )
    else:
        table.put_item(Item={"connection_id": connection_id, "timestamp": datetime_str, "lat": values[":lat"], "lon": values[":lon"], "similar_cases": values.get(":similar")})

# --- Lambda Entrypoint ---
def lambda_handler(event, context):
    route_key = event.get("requestContext", {}).get("routeKey")
    if route_key == "$connect":
        return handle_connect(event)
    elif route_key == "$disconnect":
        return handle_disconnect(event)
    elif route_key == "sendMessage":
        return handle_send_message(event)
    else:
        return {"statusCode": 400, "body": "Unsupported route"}
