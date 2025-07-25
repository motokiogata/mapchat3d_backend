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
    "Vehicle_to_vehicle_(car_accidents_against_car)": ["98-113.csv", "114-138.csv", "139-159.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_motorcycle)": ["160-204.csv", "205-234.csv"],
    "Vehicle(car_or_motorcycle)_accident_against_bicycle": ["235-280.csv", "281-310.csv"],
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

def handle_init_conversation(event):
    try:
        connection_id = event["requestContext"]["connectionId"]
        jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))

        instruction = (
            f"Your name is Mariko, and you work for Tokio Marine Nichido, an insurance company, "
            f"as a kind and helpful operator handling traffic accident claims in English.\n\n"
            f"Please start the conversation.\n"
            f"First let user to pin the drop on the map where the accident occured.\n"
        )

        chat_histories[connection_id] = [{"role": "user", "content": instruction}]

        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
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

        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"response": completion}).encode("utf-8")
        )

        return {"statusCode": 200}
    except Exception as e:
        logger.error(f"❗ initConversation failed: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

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
            f"First, please let the customer pin the drop and push the button on the map where the accident occurred on the GUI and tell the customer to go back to answer the question.\n\n"
            f"Then please ask the following two questions to gather information about the accident:\n\n"
            f"1. When did the accident occur? It is important to know the date and time of the accident. "
            f"FYI, today is \"{jst_time}\". "
            f"If the user just said an abstract time like 'today' or 'yesterday', you can guess it with the current date above."
            f"If the user said 'I don't know', confirm that they don't remember the time.\n\n"
            f"2. Where did the accident happen? Please gather detailed location information.\n\n"
            f"3. Then Please ask the customer to describe the details of the accident."
            f"Ask where their vehicle was coming from and where it was heading."
            f"Also ask the type of the other party and which direction the other party came from and headed for"
            f"If possible, gather information about the traffic signals and the speed of both vehicles, or anyother thing"
            f"Let the customer know that it's fine to share only what they can remember.\n\n"
            f"After you have received answers to those questions, The other LLM will be invoked automatically."
            f"The other LLM will ask user about the accidents' details. so don't say nothing as long as the other LLM will be satisfied.\n\n"
            f"After the other LLM has finished asking questions, Say 'I will now display the accident description below. Please wait a moment.'\n\n"
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
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
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

        similar_cases = None
        if should_run_similarity_search(full_context):
            similar_cases = find_similar_cases(full_context)

            # Collect all unique modifiers from similar cases
            modifiers = set()
            for case in similar_cases:
                modifiers.update(case.get("modifications", []))

            modifiers = list(modifiers)[:5]  # Limit to top 5 for sanity

            # Generate user-friendly questions based on modifiers
            if modifiers:
                apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
                apig.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({
                        "response": "Thanks. I'm analyzing your accident and comparing with past cases. Please wait a moment..."
                    }).encode("utf-8")
                )
                try:
                    mod_questions = generate_modifier_questions(modifiers)
                except Exception as e:
                    logger.warning(f"❗ Failed to generate modifier questions: {e}")
                    mod_questions = []

                if mod_questions:
                    # Store in session (optional)
                    # chat_sessions[connection_id]["mod_questions"] = mod_questions

                    # Build follow-up message
                    followup_msg = (
                        "Thank you. Based on similar accident cases, I have a few more questions to clarify the situation:\n\n"
                        + "\n".join(f"- {q}" for q in mod_questions)
                        + "\n\nYou can answer as much as you remember."
                    )

                    combined_reply = completion + "\n\n" + followup_msg
                    chat_histories[connection_id].append({"role": "assistant", "content": combined_reply})
                    completion = combined_reply  # ← overwrite for final post

        if datetime_str != "unknown" and location_data:
            store_to_dynamodb(connection_id, datetime_str, location_data, similar_cases)

        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"response": completion}).encode("utf-8")
        )

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"❗ Unhandled error: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# --- Helpers ---
def generate_modifier_questions(modifiers: list[str]) -> list[str]:
    prompt = (
        "You are Mariko, an empathetic insurance claim operator for Tokio Marine Nichido. "
        "Given the following internal legal or traffic modifiers that affect accident fault ratio, "
        "please turn each into a clear and friendly question you can ask a customer involved in the accident. "
        "The goal is to confirm whether each modifier applies or not.\n\n"
        "Example modifier: 'A's 30km+ speed violation +20 to A'\n"
        "Example question: 'Were you driving more than 30km/h over the speed limit at the time of the accident?'\n\n"
        f"Modifiers:\n" + "\n".join(f"- {m}" for m in modifiers[:5]) +
        "\n\nOutput ONLY the list of questions, each on a new line. No explanations."
    )

    response = bedrock.invoke_model(
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    raw_text = json.loads(response["body"].read())["content"][0]["text"]
    questions = [line.strip("- ").strip() for line in raw_text.strip().split("\n") if line.strip()]
    return questions




def extract_datetime(full_context):
    prompt = f"Generate the exact datetime (format: yyyy/mm/dd hh:mm) from this accident description: \"{full_context}\". If the user provides a relative date expression such as yesterday, today, or three days ago, instead of a specific date, estimate the exact date by refferring today's date. Just output yyyy/mm/dd hh:mm directly and Don't include any other messages. If not you can't do it, return 'unknown'."
    response = bedrock.invoke_model(
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
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
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
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
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
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
    return reply == "YES"

# --- CSV Similarity Logic ---
def categorize_accident_type(full_context):
    prompt = (
        f"Categorize this accident description into one of the following categories:\n"
        f"{list(CATEGORY_CSV_MAPPING.keys())}\n"
        f"Description:\n\"{full_context}\"\n"
        f"Reply only the category string exactly as above."
    )
    response = bedrock.invoke_model(
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    category = json.loads(response["body"].read())["content"][0]["text"].strip()
    logger.info(f"Accident category: {category}")
    return category

def load_csv_from_s3(csv_filename):
    response = s3.get_object(Bucket=CSV_BUCKET, Key=csv_filename)
    return pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

def format_cases_for_analysis(cases):
    return "\n\n".join([
        f"Case {c['case_number']}:\n"
        f"road_infrastructure: {c['road_infrastructure']}\n"
        f"traffic_control_systems: {c['traffic_control_systems']}\n"
        f"vehicle_information: {c['vehicle_information']}\n"
        f"basic_fault_ratio: {c['basic_fault_ratio']}\n"
        f"key_modifiers: {c['key_modifiers']}\n"
        f"Source File: {c['source_file']}"
        for c in cases
    ])

def find_matching_pattern(category, full_context):
    all_cases = []
    for file in CATEGORY_CSV_MAPPING.get(category, []):
        df = load_csv_from_s3(file)
        for _, row in df.iterrows():
            all_cases.append({
                "case_number": row.get("Case Number"),
                "road_infrastructure": row.get("Road Infrastructure"),
                "traffic_control_systems": row.get("Traffic Control Systems"),
                "vehicle_information": row.get("Vehicle Information"),
                "basic_fault_ratio": row.get("Basic Fault Ratio"),
                "key_modifiers": row.get("Key Modifiers"),
                "source_file": file
            })

    # Prepare prompt with formatted cases + user context
    cases_text = format_cases_for_analysis(all_cases)
    prompt = (
        f"Given the following accident cases:\n{cases_text}\n\n"
        f"Based on the accident description:\n{full_context}\n"
        f"Identify the TOP 5 MOST SIMILAR cases from the list above.\n"
        f"For each similar case, provide a JSON object with the following keys:\n"
        f"case_number, confidence_score (0-10), reasoning_details, applicable_modifications (list).\n"
        f"Output a JSON array of 5 objects, sorted by descending confidence_score.\n"
        f"Output ONLY a plain JSON array.Do not wrap it in Markdown or any other formatting like ```json."
    )

    response = bedrock.invoke_model(
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    result_text = json.loads(response["body"].read())["content"][0]["text"]
    logger.info(f"Similarity match response: {result_text}")

    json_match = re.search(r"\[\s*\{.*?\}\s*\]", result_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"Failed to parse similarity JSON: {e}")
            logger.warning(f"Claude output: {result_text}")
    return None

def calculate_final_fault_ratio(base_ratio, modifications, accident_context):
    prompt = (
        f"Given a base fault ratio: {base_ratio}, and the following modification factors: {modifications},\n"
        f"and the accident description:\n{accident_context}\n"
        f"Calculate the final fault ratio as a numeric value between 0 and 1.\n"
        f"Output only the numeric value as a string."
    )
    response = bedrock.invoke_model(
        modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    ratio_text = json.loads(response["body"].read())["content"][0]["text"].strip()
    logger.info(f"Calculated final fault ratio: {ratio_text}")
    return ratio_text

def find_similar_cases(full_context):
    category = categorize_accident_type(full_context)
    matches = find_matching_pattern(category, full_context)  # should return a list of 5 dicts

    if matches:
        results = []
        for match in matches:
            final_fault_ratio = calculate_final_fault_ratio(
                match.get("fault_ratio", "unknown"),
                match.get("applicable_modifications", []),
                full_context
            )
            results.append({
                "case_id": match.get("case_number"),
                "category": category,
                "similarity": float(match.get("confidence_score", 0)) / 10.0,
                "summary": match.get("reasoning_details", ""),
                "fault_ratio": final_fault_ratio,
                "base_fault_ratio": match.get("fault_ratio", "unknown"),
                "modifications": match.get("applicable_modifications", [])
            })
        return results
    return None

from decimal import Decimal

def store_to_dynamodb(connection_id, datetime_str, location, similar_cases=None):
    item = table.get_item(Key={"connection_id": connection_id}).get("Item")
    
    similar_data = None
    if similar_cases:
        similar_data = [{
            "case_id": case["case_id"],
            "category": case.get("category", "unknown"),
            "similarity": Decimal(str(case["similarity"])),
            "summary": case["summary"][:500],  # truncate to avoid size limit
            "fault_ratio": case.get("fault_ratio", "unknown"),
            "base_fault_ratio": case.get("base_fault_ratio", "unknown"),
            "modifications": case.get("modifications", [])[:5]
        } for case in similar_cases[:5]]  # ← store top 5

    if item:
        # Update existing record
        update_expr = "SET #ts = :ts, lat = :lat, lon = :lon"
        expr_attr_names = {"#ts": "timestamp"}
        expr_attr_values = {
            ":ts": datetime_str,
            ":lat": Decimal(str(location["lat"])),
            ":lon": Decimal(str(location["lon"]))
        }
        if similar_data:
            update_expr += ", similar_cases = :similar"
            expr_attr_values[":similar"] = similar_data

        table.update_item(
            Key={"connection_id": connection_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_attr_names,
            ExpressionAttributeValues=expr_attr_values
        )
    else:
        # Create new record
        new_item = {
            "connection_id": connection_id,
            "timestamp": datetime_str,
            "lat": Decimal(str(location["lat"])),
            "lon": Decimal(str(location["lon"]))
        }
        if similar_data:
            new_item["similar_cases"] = similar_data

        table.put_item(Item=new_item)


# --- Lambda Entrypoint ---
def lambda_handler(event, context):
    logger.info("📦 FULL EVENT: " + json.dumps(event))
    route_key = event.get("requestContext", {}).get("routeKey")
    logger.info(f"🧭 RouteKey: {event.get('requestContext', {}).get('routeKey')}")
    if route_key == "$connect":
        return handle_connect(event)
    elif route_key == "$disconnect":
        return handle_disconnect(event)
    elif route_key == "sendMessage":
        return handle_send_message(event)
    elif route_key == "initConversation":
        return handle_init_conversation(event)  # ← add this
    else:
        return {"statusCode": 400, "body": "Unsupported route"}