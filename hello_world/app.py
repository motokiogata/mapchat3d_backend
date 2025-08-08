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
import time

# --- Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table("AccidentDataTable")
s3 = boto3.client("s3")
lambda_client = boto3.client('lambda')  # ← ADD THIS LINE
CSV_BUCKET = os.environ.get('CSV_BUCKET_NAME', 'your-csv-bucket')
BUCKET = os.environ.get("FIELD_OUTPUT_BUCKET", "your-default-bucket-name")
ANALYTICS_FUNCTION_NAME = os.environ.get('ANALYTICS_FUNCTION_NAME')  # ← ADD THIS LINE

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
            Data=json.dumps({"response": completion, "connectionId": connection_id}).encode("utf-8")
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

        # ✅ ISSUE A FIX: Handle location detection message
        if user_msg.strip() == "Location was detected. The map analytics will be proccessed.":
            return handle_location_detected(connection_id, event, chat_histories)

        # ✅ ISSUE B FIX: Handle analytics completion message  
        if user_msg.strip() == "Processing complete. The map data is ready for LLM":
            analytics_result = handle_analytics_trigger(connection_id, event, chat_histories)
            
            if analytics_result.get("continue_conversation"):
                return {"statusCode": 200}
            else:
                return analytics_result

        logger.info(f"📜 Chat history for {connection_id}: {chat_histories[connection_id]}")

        # Continue with normal Claude processing...
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

        # Rest of your existing logic (datetime, location extraction, similarity search, etc.)
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[connection_id]])
        datetime_str = extract_datetime(full_context)
        location_data = extract_location(full_context)

        similar_cases = None
        if should_run_similarity_search(full_context):
            similar_cases = find_similar_cases(full_context)

            modifiers = set()
            for case in similar_cases:
                modifiers.update(case.get("modifications", []))
            modifiers = list(modifiers)[:5]

            if modifiers:
                apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
                apig.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({
                        "response": "Thanks. I'm analyzing your accident and comparing with past cases. Please wait a moment...",
                        "connectionId": connection_id
                    }).encode("utf-8")
                )
                try:
                    mod_questions = generate_modifier_questions(modifiers)
                except Exception as e:
                    logger.warning(f"❗ Failed to generate modifier questions: {e}")
                    mod_questions = []

                if mod_questions:
                    followup_msg = (
                        "Thank you. Based on similar accident cases, I have a few more questions to clarify the situation:\n\n"
                        + "\n".join(f"- {q}" for q in mod_questions)
                        + "\n\nYou can answer as much as you remember."
                    )
                    combined_reply = completion + "\n\n" + followup_msg
                    chat_histories[connection_id].append({"role": "assistant", "content": combined_reply})
                    completion = combined_reply

        if datetime_str != "unknown" and location_data:
            store_to_dynamodb(connection_id, datetime_str, location_data, similar_cases)

        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"response": completion, "connectionId": connection_id}).encode("utf-8")
        )

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"❗ Unhandled error: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# --- Helpers ---
def handle_location_detected(connection_id, event, chat_histories):
    """
    Handle when location is detected - Issue A fix
    LLM should acknowledge and say it will collect location info
    """
    logger.info(f"📍 Location detected for connection: {connection_id}")
    
    try:
        # Create specific instruction for location detection
        location_instruction = """
        The customer has just pinned a location on the map where their accident occurred. 
        The system is now processing the map data to gather information about the road infrastructure, 
        traffic signals, intersections, and other relevant details at that location.
        
        Please acknowledge this and let the customer know you're analyzing the location. 
        You should say something like:
        "Thank you for pinning the location. I'll collect information about the road layout, 
        traffic signals, and intersection details at that location. Please wait a moment while 
        I analyze the map data..."
        
        Do NOT ask about accident details yet. Just acknowledge and wait.
        """
        
        chat_histories[connection_id].append({"role": "user", "content": location_instruction})
        
        # Let Claude generate appropriate response
        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": chat_histories[connection_id]
            })
        )

        completion = json.loads(response["body"].read())["content"][0]["text"]
        chat_histories[connection_id].append({"role": "assistant", "content": completion})

        # Send response to user
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": completion,
                "connectionId": connection_id,
                "type": "location_acknowledged"
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Failed to handle location detection: {e}")
        error_msg = "Thank you for pinning the location. I'm now analyzing the area. Please wait a moment..."
        chat_histories[connection_id].append({"role": "assistant", "content": error_msg})
        
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": error_msg,
                "connectionId": connection_id
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}

def handle_analytics_trigger(connection_id, event, chat_histories):
    """
    Handle the analytics trigger message and process results - Issue B fix
    LLM should explain what it sees and ask specific questions
    """
    logger.info(f"🚀 Triggering analytics for connection: {connection_id}")
    
    try:
        # Send immediate response
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "Perfect! I've received the map analysis. Let me review what I can see about the accident location...",
                "connectionId": connection_id
            }).encode("utf-8")
        )

        # Invoke analytics function SYNCHRONOUSLY to get results
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'chat_history': chat_histories[connection_id]  # Pass chat context
            })
        )
        
        # Parse the analytics result
        result_payload = json.loads(response['Payload'].read())
        analytics_report = result_payload.get('analytics_report', 'Analytics completed but no report generated.')
        
        # Add analytics report to chat history as system context
        analytics_context = f"[ANALYTICS RESULTS]\n{analytics_report}\n[END ANALYTICS]"
        chat_histories[connection_id].append({"role": "user", "content": analytics_context})
        
        # ✅ ISSUE B FIX: Better instruction for sharing what Mariko sees
        followup_instruction = f"""
        Based on the traffic infrastructure analysis above, please:
        
        1. **First, explain what you can see about the location** in a conversational way:
           - "Looking at the map analysis, I can see this accident occurred at [describe intersection/road type]"
           - "The analysis shows [road names/route numbers if available]"
           - "I can see there are [traffic signals/stop signs/road configuration]"
           
        2. **Then ask specific, targeted questions** based on what you observed:
           - About their route: "I can see this is where [road A] meets [road B]. Which direction were you coming from?"
           - About traffic controls: "The analysis shows there's a traffic signal here. What color was it when you approached?"
           - About their destination: "And where were you heading - which direction were you planning to turn?"
        
        3. **Make it conversational and empathetic**:
           - Show that you understand the location
           - Ask 2-3 specific questions based on the infrastructure findings
           - Let them know it's OK to share what they remember
        
        The goal is to make the customer feel like you really understand their accident location and are asking informed questions.
        """
        
        chat_histories[connection_id].append({"role": "user", "content": followup_instruction})
        
        # Let Claude generate response with analytics context
        claude_response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "temperature": 0.1,
                "messages": chat_histories[connection_id]
            })
        )

        completion = json.loads(claude_response["body"].read())["content"][0]["text"]
        chat_histories[connection_id].append({"role": "assistant", "content": completion})

        # Send Claude's response with analytics-based questions
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": completion,
                "connectionId": connection_id,
                "type": "analytics_complete"
            }).encode("utf-8")
        )
        
        return {
            "statusCode": 200, 
            "continue_conversation": True,
            "analytics_completed": True
        }
        
    except Exception as e:
        logger.error(f"❗ Failed to process analytics: {e}")
        error_msg = "I can see the location analysis, but let me ask you some questions about what happened. Can you tell me which direction you were coming from and where you were heading?"
        chat_histories[connection_id].append({"role": "assistant", "content": error_msg})
        
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": error_msg,
                "connectionId": connection_id
            }).encode("utf-8")
        )
        
        return {
            "statusCode": 200, 
            "continue_conversation": True,
            "error": True
        }



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


def handle_check_processing_done(event):
    try:
        params = event.get("queryStringParameters", {})
        connection_id = params.get("connectionId")
        if not connection_id:
            return {"statusCode": 400, "body": "Missing connectionId parameter"}

        logger.info(f"🔍 Polling for done.flag for {connection_id}")
        key = f"outputs/{connection_id}/done.flag"
        found = False
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            found = True
        except s3.exceptions.ClientError:
            found = False

        return {
            "statusCode": 200,
            "body": json.dumps({"status": "done" if found else "processing"})
        }

    except Exception as e:
        logger.error(f"❌ Error in handle_check_processing_done: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# --- Lambda Entrypoint ---
def lambda_handler(event, context):
    logger.info("📦 FULL EVENT: " + json.dumps(event))
    route_key = event.get("requestContext", {}).get("routeKey")
    logger.info(f"🧭 RouteKey: {event.get('requestContext', {}).get('routeKey')}")
    if route_key:
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
    
    # REST (polling)
    path = event.get("resource")
    http_method = event.get("httpMethod")
    logger.info(f"🌐 REST API Request: {http_method} {path}")
    if http_method == "GET" and path == "/checkProcessingDone":
        return handle_check_processing_done(event)
    else:
        return {"statusCode": 400, "body": "Unsupported request"}