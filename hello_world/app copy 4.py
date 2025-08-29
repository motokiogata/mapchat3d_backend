#app.py
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
import threading

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
def clear_interactive_state(connection_id):
    """Clear interactive analytics state"""
    try:
        state_key = f"interactive-state/{connection_id}/state.json"
        s3.delete_object(Bucket=BUCKET, Key=state_key)
        logger.info(f"🧹 Cleared interactive state for {connection_id}")
    except Exception as e:
        logger.error(f"❗ Failed to clear interactive state: {e}")

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
            f"Please start the conversation. You DON'T NEED TO ASK the user whether they're safe or have called the police or any emergency staff because it's already done and over by human operator and you can just focus on how the traffic accident happened.\n"
            f"First let user to pin the drop on the map where the accident occured.\n"
            f"Precise Example is like this: Hello! My name is Mariko, and I'm here to assist you with your traffic accident claim today. I'm so sorry to hear that you've been involved in an accident, and I want to help make this process as smooth as possible for you. To get started with your claim, I'll need to gather some important information about the incident. First, could you please help me locate exactly where the accident occurred? I'm going to share a map with you, and I'd like you to drop a pin at the precise location where the accident took place. This will help us process your claim more efficiently and ensure we have accurate location details for our records. Please take your time to find the exact spot on the map. If you need any assistance with using the map feature or have any questions, please don't hesitate to ask me. I'm here to help you through every step of this process. *[Waiting for you to place the pin on the map]*"
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

        # Check if in interactive analytics mode for Analytics Phase 2
        interactive_state = get_interactive_state(connection_id)
        if interactive_state and interactive_state["mode"] == "interactive_analytics":
            logger.info(f"🕵️ User in interactive analytics mode")
            return handle_interactive_analytics_response(connection_id, user_msg, interactive_state["conversation_state"], event)


        instruction = (
            f"Your name is Mariko, and you work for Tokio Marine Nichido, an insurance company, "
            f"as a kind and helpful operator handling traffic accident claims in English.\n\n"
            
            f"🚨 CRITICAL RULE: Ask ONLY ONE question per response. NEVER ask multiple questions. STOP after asking one question and wait for the user's answer.\n\n"
            
            f"STEP 1: After the customer pins the location, ask ONLY this: "
            f"'Thank you for providing the location. Could you please tell me what date and time the accident occurred?'\n"
            f"TODAY IS: {jst_time}\n"
            f"STOP HERE. Wait for their response.\n\n"
            
            f"STEP 2: After they answer about time, ask ONLY this: "
            f"'Thank you. Now, could you please describe what happened during the accident? Take your time and share what you remember.'\n"
            f"STOP HERE. Wait for their response.\n\n"
            
            f"STEP 3: Based on their description, ask follow-up questions ONE BY ONE:\n"
            f"- Ask about collision type\n"
            f"- Ask about vehicles involved  \n"
            f"- Ask about traffic conditions\n"
            f"ONE QUESTION PER RESPONSE ONLY.\n\n"
            
            f"❌ DO NOT:\n"
            f"- Ask multiple questions in one response\n"
            f"- List questions with bullet points\n"
            f"- Say 'I'd like to know' followed by a list\n"
            f"- Use phrases like 'Could you tell me about A, B, and C'\n\n"
            
            f"✅ DO:\n"
            f"- Ask one specific question\n"
            f"- Wait for response\n"
            f"- Then ask the next question\n\n"
            
            f"If you ask more than one question in a single response, you have failed."
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
                "max_tokens": 10000,
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

###############

def check_analytics_result(connection_id):
    result_key = f"analytics-results/{connection_id}/result.json"
    logger.info(f"🔍 Checking S3 for key: {result_key}")
    logger.info(f"🔍 Using bucket: {BUCKET}")
    
    try:
        # Try to list objects in the directory first
        list_response = s3.list_objects_v2(
            Bucket=BUCKET, 
            Prefix=f"analytics-results/{connection_id}/"
        )
        
        if 'Contents' in list_response:
            logger.info(f"📁 Files found in analytics-results/{connection_id}/:")
            for obj in list_response['Contents']:
                logger.info(f"📄 File: {obj['Key']}, Size: {obj['Size']}, Modified: {obj['LastModified']}")
        else:
            logger.info(f"📁 No files found in analytics-results/{connection_id}/")
        
        # Now try to get the specific file
        response = s3.get_object(Bucket=BUCKET, Key=result_key)
        logger.info(f"✅ Found analytics result in S3!")
        
        result_data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"📄 Result data keys: {list(result_data.keys())}")
        
        # Don't delete yet - let's confirm it works first
        # s3.delete_object(Bucket=BUCKET, Key=result_key)
        
        return result_data
    except s3.exceptions.NoSuchKey:
        logger.info(f"❌ NoSuchKey: {result_key}")
        return None
    except Exception as e:
        logger.error(f"❌ Error checking analytics result: {e}")
        return None


# Add new handler for interactive responses:
def handle_interactive_analytics_response(connection_id, user_input, current_state, event):
    """Handle user responses during interactive investigation"""
    try:
        logger.info(f"🤖 Processing analytics response: '{user_input}'")
        
        # Call your existing analytics function
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'user_response',  # ✅ Analytics already handles this
                'user_input': user_input,
                'conversation_state': current_state
            })
        )
        
        result = json.loads(response['Payload'].read())
        
        if result.get("statusCode") == 200:
            # Update state
            new_state = result.get("conversation_state")
            if new_state:
                store_interactive_state(connection_id, new_state)
            
            # Send response
            apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
            message = result.get("message", "Thank you.")
            
            # Add to chat history
            if connection_id in chat_histories:
                chat_histories[connection_id].append({"role": "user", "content": user_input})
                chat_histories[connection_id].append({"role": "assistant", "content": message})
            
            # Check if complete
            if result.get("animation_ready"):
                message += "\n\n✅ Investigation Complete!"
                clear_interactive_state(connection_id)
            
            apig.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "response": message,
                    "connectionId": connection_id
                }).encode("utf-8")
            )
            
            return {"statusCode": 200}
        else:
            raise Exception(f"Analytics error: {result}")
            
    except Exception as e:
        logger.error(f"❗ Interactive analytics failed: {e}")
        clear_interactive_state(connection_id)
        
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "Let me continue with some general questions about your accident.",
                "connectionId": connection_id
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}

def start_interactive_investigation(connection_id, event, chat_histories, analytics_result, apig):
    """Start the interactive investigation phase"""
    try:
        logger.info("🕵️ Starting interactive investigation phase")
        
        # Get the first investigative question from analytics result
        first_question = analytics_result.get('message', 'Let me ask you about the accident...')
        conversation_state = analytics_result.get('conversation_state', 'awaiting_direction')
        
        # ✅ ADD ANALYTICS CONTEXT AS ASSISTANT MESSAGE
        scene_analysis = analytics_result.get('scene_analysis', 'Scene analysis completed.')
        analytics_context = f"[INFRASTRUCTURE ANALYSIS COMPLETE]\n{scene_analysis}\n[STARTING INTERACTIVE INVESTIGATION]"
        chat_histories[connection_id].append({"role": "assistant", "content": analytics_context})
        
        # ✅ ADD FIRST QUESTION TO CHAT HISTORY
        chat_histories[connection_id].append({"role": "assistant", "content": first_question})
        
        # ✅ MARK THIS CONNECTION AS IN INTERACTIVE MODE
        store_interactive_state(connection_id, conversation_state)
        
        # ✅ SEND FIRST INVESTIGATIVE QUESTION TO USER
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": first_question,
                "connectionId": connection_id,
                "type": "interactive_investigation_start",
                "conversation_state": conversation_state
            }).encode("utf-8")
        )
        
        logger.info(f"✅ Interactive investigation started for {connection_id}")
        return {"statusCode": 200, "continue_conversation": True}
        
    except Exception as e:
        logger.error(f"❗ Failed to start interactive investigation: {e}")
        
        # Fallback to basic question
        fallback_msg = "I can see the location details. Could you tell me which direction you were coming from when the accident happened?"
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": fallback_msg,
                "connectionId": connection_id,
                "type": "investigation_fallback"
            }).encode("utf-8")
        )
        
        return {"statusCode": 200, "continue_conversation": True}



def store_interactive_state(connection_id, conversation_state):
    """Store that this connection is in interactive analytics mode"""
    try:
        state_data = {
            "mode": "interactive_analytics",
            "conversation_state": conversation_state,
            "timestamp": datetime.now().isoformat()
        }
        
        state_key = f"interactive-state/{connection_id}/state.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=state_key,
            Body=json.dumps(state_data),
            ContentType='application/json'
        )
        logger.info(f"💾 Interactive state stored: {state_key}")
    except Exception as e:
        logger.error(f"❗ Failed to store interactive state: {e}")


def get_interactive_state(connection_id):
    """Get interactive state for a connection"""
    try:
        state_key = f"interactive-state/{connection_id}/state.json"
        response = s3.get_object(Bucket=BUCKET, Key=state_key)
        state_data = json.loads(response['Body'].read().decode('utf-8'))
        return state_data
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error(f"❗ Failed to get interactive state: {e}")
        return None
        
        
def handle_analytics_trigger(connection_id, event, chat_histories):
    """
    Handle analytics - invoke async for initial analysis, then become interactive
    """
    logger.info(f"🚀 Triggering analytics for connection: {connection_id}")
    
    try:
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        
        # Send immediate response
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "Perfect! I've received the map analysis. Let me review what I can see about the accident location...",
                "connectionId": connection_id
            }).encode("utf-8")
        )

        # ✅ INVOKE ANALYTICS FOR INITIAL ANALYSIS
        logger.info(f"🚀 Invoking analytics function for initial analysis: {ANALYTICS_FUNCTION_NAME}")
        
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='Event',  # ← Async invoke for heavy initial analysis
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'initial_analysis',  # ← NEW: Specify this is initial analysis
                'chat_history': chat_histories[connection_id]
            })
        )
        
        logger.info(f"✅ Initial analytics invoked! StatusCode: {response.get('StatusCode')}")

        # ✅ POLL FOR INITIAL ANALYSIS COMPLETION
        logger.info("🔄 Starting polling for initial analysis completion...")
        
        max_wait_time = 240  # 4 minutes for heavy analysis
        check_interval = 18
        elapsed_time = 0
        
        progress_messages = [
            "Analyzing intersection infrastructure and traffic controls...",
            "Processing road layout and lane configurations...",
            "Examining traffic signal patterns and right-of-way rules...",
            "Building comprehensive scene understanding...",
            "Preparing interactive investigation questions...",
            "Almost ready to start the detailed investigation...",
            "Finalizing traffic infrastructure analysis..."
        ]
        
        message_index = 0
        
        while elapsed_time < max_wait_time:
            logger.info(f"🔁 Polling iteration: elapsed_time={elapsed_time}")
            
            try:
                # Send progress message
                if message_index < len(progress_messages):
                    logger.info(f"📤 Sending progress message: {progress_messages[message_index]}")
                    apig.post_to_connection(
                        ConnectionId=connection_id,
                        Data=json.dumps({
                            "response": progress_messages[message_index],
                            "connectionId": connection_id,
                            "type": "analysis_progress"
                        }).encode("utf-8")
                    )
                    message_index += 1
                
                # Check if initial analysis is complete
                logger.info(f"🔍 Checking for initial analysis result...")
                analytics_result = check_initial_analysis_result(connection_id)
                
                if analytics_result:
                    logger.info("🎯 Initial analysis complete! Starting interactive mode...")
                    return start_interactive_investigation(connection_id, event, chat_histories, analytics_result, apig)
                else:
                    logger.info("📄 Initial analysis not ready yet")
                
                # Wait before next check
                time.sleep(check_interval)
                elapsed_time += check_interval
                
            except Exception as poll_error:
                logger.error(f"❌ Error in polling loop: {poll_error}")
                break
        
        # If we get here, initial analysis took too long
        logger.warning("⏰ Initial analysis timeout")
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "The detailed analysis is taking longer than expected. Let me ask you some basic questions while it completes...",
                "connectionId": connection_id,
                "type": "analysis_timeout"
            }).encode("utf-8")
        )
        
        return {"statusCode": 200, "continue_conversation": True}
        
    except Exception as e:
        logger.error(f"❗ Failed to process analytics: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}


def handle_analytics_trigger(connection_id, event, chat_histories):
    """Handle analytics - call synchronously and get immediate response"""
    logger.info(f"🚀 Triggering analytics for connection: {connection_id}")
    
    try:
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        
        # Send immediate response
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "Perfect! Let me analyze the location details and start asking specific questions...",
                "connectionId": connection_id
            }).encode("utf-8")
        )

        # ✅ CALL ANALYTICS SYNCHRONOUSLY
        logger.info(f"🚀 Calling analytics function synchronously: {ANALYTICS_FUNCTION_NAME}")
        
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',  # ← Synchronous!
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'initial_analysis'
            })
        )
        
        result = json.loads(response['Payload'].read())
        logger.info(f"📊 Analytics result: {result}")
        
        if result.get("statusCode") == 200:
            # ✅ IMMEDIATELY START INTERACTIVE INVESTIGATION
            return start_interactive_investigation(connection_id, event, chat_histories, result, apig)
        else:
            # Fallback
            apig.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "response": "I can see the location. Could you tell me which direction you were coming from when the accident happened?",
                    "connectionId": connection_id
                }).encode("utf-8")
            )
            return {"statusCode": 200, "continue_conversation": True}
        
    except Exception as e:
        logger.error(f"❗ Failed to process analytics: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}


def handle_analytics_result(connection_id, event, chat_histories, analytics_result, apig):
    """
    Process analytics result and send Claude response
    """
    try:
        analytics_report = analytics_result.get('analytics_report', 'Analytics completed.')
        
        # ✅ ADD TO CHAT HISTORY (Main function maintains this)
        analytics_context = f"[ANALYTICS RESULTS]\n{analytics_report}\n[END ANALYTICS]"
        chat_histories[connection_id].append({"role": "user", "content": analytics_context})
        
        # ✅ GENERATE CLAUDE RESPONSE (Main function does this)
        followup_instruction = """
        Based on the traffic infrastructure analysis above, please:
        
        1. First, explain what you can see about the location in a conversational way
        2. Then ask specific, targeted questions based on what you observed  
        3. Make it conversational and empathetic
        
        The goal is to make the customer feel like you understand their accident location.
        """
        
        chat_histories[connection_id].append({"role": "user", "content": followup_instruction})
        
        # Generate Claude response
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

        # ✅ SEND FINAL RESPONSE FROM MAIN FUNCTION
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": completion,
                "connectionId": connection_id,
                "type": "analytics_complete"
            }).encode("utf-8")
        )
        
        logger.info(f"✅ Analytics result processed and sent to user")
        return {"statusCode": 200, "continue_conversation": True}
        
    except Exception as e:
        logger.error(f"❗ Failed to process analytics result: {e}")
        
        error_msg = "I can see the location analysis, but let me ask you some questions about what happened. Can you tell me which direction you were coming from?"
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": error_msg,
                "connectionId": connection_id,
                "type": "error"
            }).encode("utf-8")
        )
        
        return {"statusCode": 500, "body": "Error processing analytics result"}

##############

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

#for confirming the action is "ping" in $default route
def _parse_body(event):
    """Parse the body from WebSocket event"""
    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            return json.loads(body)
        return body
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse body: {e}")
        return {}

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
        elif route_key == "$default":
            body = _parse_body(event)
            if body.get("action") == "ping":
                # One-way keep-alive; no pong needed
                return {"statusCode": 200}
            # (Optional) quietly accept other unknown actions to avoid drops
            logger.warning(f"Unknown action via $default: {body}")
            return {"statusCode": 200}
        else:
            # With $default defined, this 'else' should rarely hit.
            return {"statusCode": 400, "body": "Unsupported route"}

    # REST (polling)
    path = event.get("resource")
    http_method = event.get("httpMethod")
    logger.info(f"🌐 REST API Request: {http_method} {path}")
    if http_method == "GET" and path == "/checkProcessingDone":
        return handle_check_processing_done(event)
    else:
        return {"statusCode": 400, "body": "Unsupported request"}