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
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

LANG_CODES = {"EN","JA","ES","KO","ZH"}

def detect_language_via_llm(user_text: str) -> str:
    """
    Call LLM once to classify language.
    Returns one of: EN, JA, ES, KO, ZH, UNKNOWN
    """
    try:
        instruction = (
            "You are a strict language identifier.\n"
            "Your only job is to read the user's message and respond with exactly one of:\n"
            "EN, JA, ES, KO, ZH, UNKNOWN\n"
            "No explanation. No punctuation. No extra words."
            "Classify the language of the following user message.\n"
            f"User message:\n<<<\n{user_text}\n>>>\n"
        )

        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": instruction},
                ]
            })
        )
        text = json.loads(response["body"].read())["content"][0]["text"].strip().upper()
        return text if text in LANG_CODES or text == "UNKNOWN" else "UNKNOWN"
    except Exception as e:
        logger.error(f"❗ language detect error: {e}")
        return "UNKNOWN"


def normalize_lang(user_text: str) -> str:
    """Soft, forgiving mapping. Returns 'en','ja','es','ko','zh' or '' if unsure."""
    if not user_text:
        return ""
    t = user_text.strip().lower()

    # direct codes
    if t in {"en","ja","es","ko","zh"}:
        return t

    # look for code inside longer sentence
    for code in ["en","ja","es","ko","zh"]:
        if f" {code} " in f" {t} ":
            return code

    # language names / native scripts
    if any(k in t for k in ["english","inglés"]): return "en"
    if any(k in t for k in ["日本語","japanese","japonés","japones","jp"]): return "ja"
    if any(k in t for k in ["español","spanish","espanol"]): return "es"
    if any(k in t for k in ["한국어","korean","coreano","kr"]): return "ko"
    if any(k in t for k in ["中文","chinese","chino","cn","汉语","漢語"]): return "zh"

    # final heuristic: if the text contains many CJK chars, guess zh/ja/ko
    cjk = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff')
    hangul = sum(1 for ch in t if '\uac00' <= ch <= '\ud7af')
    if hangul > 0: return "ko"
    if cjk > 0:
        # very rough: presence of "の/です/ます/に/を" may hint Japanese
        if any(k in t for k in ["です","ます","の","に","を","が"]): return "ja"
        return "zh"

    return ""  # let the system ask again


# --- Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
region = os.environ.get("AWS_REGION", "us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name=region)
dynamodb = boto3.resource("dynamodb", region_name=region)
table = dynamodb.Table("AccidentDataTable")
s3 = boto3.client("s3")
lambda_client = boto3.client('lambda')
CSV_BUCKET = os.environ.get('CSV_BUCKET_NAME', 'your-csv-bucket')
BUCKET = os.environ.get("FIELD_OUTPUT_BUCKET", "your-default-bucket-name")
ANALYTICS_FUNCTION_NAME = os.environ.get('ANALYTICS_FUNCTION_NAME')

# Global state
chat_histories = {}
apig_clients = {}



# --- Task Orchestrator ---
class ConversationPhase(Enum):
    GREETING = "greeting"
    LANGUAGE_SELECTION = "language_selection"
    LOCATION_PINNING = "location_pinning"
    PARALLEL_INFO_GATHERING = "parallel_info_gathering"
    LOCATION_PROCESSING = "location_processing"
    BASIC_INFO_GATHERING = "basic_info_gathering"
    ANALYTICS_PROCESSING = "analytics_processing"
    DETAILED_INVESTIGATION = "detailed_investigation"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    LEGAL_MODIFIER_INVESTIGATION = "legal_modifier_investigation"  # NEW
    CASE_FINALIZATION = "case_finalization"
    COMPLETED = "completed"

@dataclass
class TaskState:
    phase: ConversationPhase
    sub_step: int = 0
    collected_data: Dict[str, Any] = None
    pending_questions: List[str] = None
    analytics_result: Dict[str, Any] = None
    location_attempts: int = 0
    map_processing_started: bool = False  # NEW: Track if map processing started
    questions_asked: Dict[str, bool] = None  # Track which questions have been 
    user_language: str = None  # instead of "en"
    awaiting_user_input_for: str = None  # NEW: track what we're waiting for
    modifier_questions: List[str] = None
    modifier_answers: Dict[str, str] = None
    current_modifier_index: int = 0
    #user_language: str = "en"  # NEW: default to English
    
    def __post_init__(self):
        if self.collected_data is None:
            self.collected_data = {}
        if self.pending_questions is None:
            self.pending_questions = []
        if self.questions_asked is None:
            self.questions_asked = {"datetime": False, "description": False}
        if self.modifier_questions is None:
            self.modifier_questions = []
        if self.modifier_answers is None:
            self.modifier_answers = {}

class TaskOrchestrator:
    """Central controller for managing conversation flow and task delegation"""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.state = TaskState(phase=ConversationPhase.GREETING)
        
    def get_next_action(self, user_input: str = None) -> Dict[str, Any]:
        """Determine next action based on current phase and user input"""
        
        if self.state.phase == ConversationPhase.GREETING:
            return self._handle_greeting()

        elif self.state.phase == ConversationPhase.LANGUAGE_SELECTION:
            return self._handle_language_selection(user_input)
            
        elif self.state.phase == ConversationPhase.LOCATION_PINNING:
            return self._handle_location_pinning(user_input)
            
        elif self.state.phase == ConversationPhase.PARALLEL_INFO_GATHERING:
            return self._handle_parallel_info_gathering(user_input)
            
        elif self.state.phase == ConversationPhase.LOCATION_PROCESSING:
            return self._handle_location_processing()
            
        elif self.state.phase == ConversationPhase.BASIC_INFO_GATHERING:
            return self._handle_basic_info_gathering(user_input)
            
        elif self.state.phase == ConversationPhase.ANALYTICS_PROCESSING:
            return self._handle_analytics_processing()
            
        elif self.state.phase == ConversationPhase.DETAILED_INVESTIGATION:
            return self._handle_detailed_investigation(user_input)
            
        elif self.state.phase == ConversationPhase.SIMILARITY_ANALYSIS:
            return self._handle_similarity_analysis()

        elif self.state.phase == ConversationPhase.LEGAL_MODIFIER_INVESTIGATION:
            return self._handle_legal_modifier_investigation(user_input)
            
        elif self.state.phase == ConversationPhase.CASE_FINALIZATION:
            return self._handle_case_finalization()
            
        else:
            return {"action": "error", "message": "Unknown phase"}

    def _handle_legal_modifier_investigation(self, user_input: str) -> Dict[str, Any]:
        lang = (self.state.user_language or "en").upper()
        
        # Store user's answer if provided
        if user_input and user_input.strip() and self.state.current_modifier_index > 0:
            question_key = f"modifier_{self.state.current_modifier_index - 1}"
            self.state.modifier_answers[question_key] = user_input
        
        # Check if we have more questions to ask
        if self.state.current_modifier_index < len(self.state.modifier_questions):
            current_question = self.state.modifier_questions[self.state.current_modifier_index]
            self.state.current_modifier_index += 1
            
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    f"Ask this specific question to determine legal liability factors: "
                    f"'{current_question}' "
                    "Be empathetic and explain that this helps determine accurate fault ratios."
                ),
                "progress": f"Legal assessment (Step 6/6) - Question {self.state.current_modifier_index}/{len(self.state.modifier_questions)}"
            }
        else:
            # All modifier questions answered, move to finalization
            self.state.phase = ConversationPhase.CASE_FINALIZATION
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Thank the user for providing detailed information and explain that "
                    "you're now calculating the final fault ratio and preparing their claim summary."
                ),
                "progress": "Calculating fault ratio (Step 6/6)"
            }


    def _handle_greeting(self) -> Dict[str, Any]:
        # Move to language selection; LLM will produce a multilingual greeting on its own
        self.state.phase = ConversationPhase.LANGUAGE_SELECTION

        return {
            "action": "claude_response",
            # Keep the system light—just set role; let the model compose freely
            "system": (
                "You are Mariko, a claims operator at Tokio Marine & Nichido. "
                "Be polite, concise, and professional."
            ),
            # ✅ Your working instruction (as you wrote it)
            "instruction": (
                "あなたは東京海上日動で働いているマリコです。あなたはマルチリンガルで、"
                "日本語、英語、韓国語、中国語が話せます。まずはすべての言葉であいさつをしながら、"
                "どの言語を選ぶかお客様に聞いてください。"
            ),
            "progress": "Language selection"
        }

    def _handle_language_selection(self, user_input: str) -> Dict[str, Any]:
        code = detect_language_via_llm(user_input)  # -> EN/JA/ES/KO/ZH/UNKNOWN

        if code == "UNKNOWN":
            # Ask again in multiple languages
            return {
                "action": "claude_response",
                "instruction": (
                    "The user hasn't clearly selected a language. "
                    "Ask them again to choose their preferred language, offering: "
                    "English (EN), 日本語 (JA), Español (ES), 한국어 (KO), 中文 (ZH). "
                    "Be multilingual in your response."
                ),
                "progress": "Language selection"
            }

        # Lock language + move to next phase
        self.state.user_language = code.lower()
        self.state.phase = ConversationPhase.LOCATION_PINNING

        return {
            "action": "claude_response",
            "instruction": (
                f"Reply only in {code}. "
                "Tell the user they have a map UI and ask them to: "
                "1) Drop a pin on the exact accident location, "
                "2) Click the button below the map, "
                "3) Then wait a moment."
            ),
            "progress": "Pin the location (Step 1/6)"
        }

    def _handle_location_pinning(self, user_input: str) -> Dict[str, Any]:
        lang = (self.state.user_language or "en").upper()

        logger.info(f"🗺️ Location pinning - received input: '{user_input}'")
        location_detected = False

        if user_input:
            # Your existing heuristics for detection
            indicators = ["Location was detected", "Location detected", "coordinates", "lat", "longitude", "pinned", "dropped"]
            for indicator in indicators:
                if indicator.lower() in user_input.lower():
                    location_detected = True
                    break
            if not location_detected:
                for word in user_input.split():
                    try:
                        float(word)
                        if "." in word and len(word) > 3:
                            location_detected = True
                            break
                    except ValueError:
                        continue

        if location_detected:
            logger.info(f"✅ Locations detected for {self.connection_id}")
            # Advance phase; map analysis can run in parallel as before
            self.state.phase = ConversationPhase.PARALLEL_INFO_GATHERING
            self.state.map_processing_started = True

            # Say a short acknowledgment, then immediately proceed to ask first question
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Say this to the user, concisely, in 1–2 short sentences:\n"
                    "Thanks—I've received your pinned location. I'm analyzing the map in the background now. "
                    "Let me start with the first question: What date and time did the accident occur?"
                ),
                "progress": "Processing location & gathering info (Step 2/6)"
            }

        # Not detected yet → keep asking, up to N tries, but do NOT crash or change architectures
        self.state.location_attempts += 1
        if self.state.location_attempts > 3:
            logger.warning(f"⚠️ No pin after multiple attempts for {self.connection_id}; continuing.")
            self.state.phase = ConversationPhase.PARALLEL_INFO_GATHERING
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Say this to the user, concisely, in 1–2 short sentences:\n"
                    "“No problem—let’s continue while the map is processing. First, what date and time did the accident occur?”"
                ),
                "progress": "Gathering basic information (Step 3/6)"
            }

        # Ask again (simple, no i18n table)
        return {
            "action": "claude_response",
            "instruction": (
                f"Reply only in {lang}. "
                "Tell the user, in one short sentence: "
                "“Please use the map UI to drop a pin on the exact accident location, then press the button below the map.”"
            ),
            "progress": "Waiting for location (Step 1/6)"
        }


    def _handle_parallel_info_gathering(self, user_input: str) -> Dict[str, Any]:
        lang = (self.state.user_language or "en").upper()
        
        # Store user input if provided
        if user_input and user_input.strip():
            # Determine which question this answers
            if not self.state.questions_asked["datetime"]:
                self.state.collected_data["datetime"] = user_input
                self.state.questions_asked["datetime"] = True
            elif not self.state.questions_asked["description"]:
                self.state.collected_data["description"] = user_input
                self.state.questions_asked["description"] = True
        
        # Find next question to ask
        if not self.state.questions_asked["datetime"]:
            self.state.awaiting_user_input_for = "datetime"  # Track what we're waiting for
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Ask the user this question in a friendly, empathetic way: "
                    "'What date and time did the accident occur?'"
                ),
                "progress": "Gathering information (Step 2/6) - Question 1/2"
            }
        elif not self.state.questions_asked["description"]:
            self.state.awaiting_user_input_for = "description"  # Track what we're waiting for
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Ask the user this question in a friendly, empathetic way: "
                    "'Could you describe what happened during the accident?'"
                ),
                "progress": "Gathering information (Step 2/6) - Question 2/2"
            }
        else:
            # All questions answered, move to next phase
            self.state.phase = ConversationPhase.ANALYTICS_PROCESSING
            self.state.awaiting_user_input_for = None
            return {
                "action": "claude_response",
                "instruction": (
                    f"Reply only in {lang}. "
                    "Tell the user, concisely: "
                    "'Perfect! I have all the basic information. Now I'll analyze the location details and ask some specific questions based on the road layout.'"
                ),
                "progress": "Finalizing location analysis (Step 3/6)"
            }



    def _handle_location_processing(self) -> Dict[str, Any]:
        """Phase 3: Processing location data (now rarely used due to parallel processing)"""
        return {
            "action": "trigger_analytics",
            "message": "Analyzing road layout, traffic signals, and intersection details...",
            "progress": "Processing location data (Step 3/6)"
        }
    
    def _handle_basic_info_gathering(self, user_input: str) -> Dict[str, Any]:
        """Phase 4: Collect basic accident information - FIXED to avoid duplicate questions"""
        # Check if we already have info from parallel gathering
        if self.state.questions_asked["datetime"] and self.state.questions_asked["description"]:
            # We already have basic info, move to analytics
            self.state.phase = ConversationPhase.ANALYTICS_PROCESSING
            return self._handle_analytics_processing()
        
        basic_questions = [
            ("datetime", "What date and time did the accident occur?"),
            ("description", "Could you describe what happened during the accident?")
        ]
        
        # Store user input if provided
        if user_input and user_input.strip():
            for question_key, _ in basic_questions:
                if not self.state.questions_asked[question_key]:
                    self.state.collected_data[question_key] = user_input
                    self.state.questions_asked[question_key] = True
                    break
        
        # Find next unasked question
        next_question = None
        for question_key, question_text in basic_questions:
            if not self.state.questions_asked[question_key]:
                next_question = question_text
                break
        
        if next_question:
            return {
                "action": "claude_response",
                "instruction": f"Ask this specific question: '{next_question}'. Be conversational and empathetic. Ask ONLY this question.",
                "progress": f"Gathering basic information (Step 3/6) - Question {sum(self.state.questions_asked.values()) + 1}/{len(basic_questions)}"
            }
        else:
            # Move to analytics processing
            self.state.phase = ConversationPhase.ANALYTICS_PROCESSING
            return self._handle_analytics_processing()
    
    def _handle_analytics_processing(self) -> Dict[str, Any]:
        """Phase 5: Advanced analytics processing"""
        self.state.phase = ConversationPhase.DETAILED_INVESTIGATION
        return {
            "action": "start_detailed_analytics",
            "message": "Perfect! Now I'll analyze the specific details of your accident based on the location and your description...",
            "progress": "Running detailed analysis (Step 4/6)"
        }
    
    def _handle_detailed_investigation(self, user_input: str) -> Dict[str, Any]:
        """Phase 6: Detailed investigation with analytics"""
        return {
            "action": "interactive_investigation",
            "user_input": user_input,
            "progress": "Detailed investigation (Step 5/6)"
        }
    
    def _handle_similarity_analysis(self) -> Dict[str, Any]:
        """Phase 7: Find similar cases"""
        self.state.phase = ConversationPhase.CASE_FINALIZATION
        return {
            "action": "run_similarity_search",
            "progress": "Finding similar cases (Step 6/6)"
        }
    
    def _handle_case_finalization(self) -> Dict[str, Any]:
        """Phase 8: Finalize the case"""
        self.state.phase = ConversationPhase.COMPLETED
        return {
            "action": "finalize_case",
            "message": "Thank you for providing all the information. I'm now finalizing your claim...",
            "progress": "Finalizing claim (Complete)"
        }

# Global orchestrators
orchestrators: Dict[str, TaskOrchestrator] = {}

# --- Categories for CSV mapping ---
CATEGORY_CSV_MAPPING = {
    "Vehicle(car_or_motorcycle)_accident_against_pedestrian": ["1-25.csv", "26-50.csv"],
    "Bycicle_accident_against_pedestrian": ["51-74.csv", "75-97.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_car)": ["98-113.csv", "114-138.csv", "139-159.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_motorcycle)": ["160-204.csv", "205-234.csv"],
    "Vehicle(car_or_motorcycle)_accident_against_bicycle": ["235-280.csv", "281-310.csv"],
    "Accidents_in_highways_or_Accidents_in_park": ["311-338.csv"]
}

# --- Connection Handlers ---
def get_apig_client(domain, stage):
    endpoint = f"https://{domain}/{stage}"
    if endpoint not in apig_clients:
        apig_clients[endpoint] = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)
    return apig_clients[endpoint]

def handle_connect(event):
    logger.info(f"✅ Client connected: {event['requestContext']['connectionId']}")
    return {"statusCode": 200}

def handle_disconnect(event):
    connection_id = event['requestContext']['connectionId']
    # Clean up
    if connection_id in orchestrators:
        del orchestrators[connection_id]
    if connection_id in chat_histories:
        del chat_histories[connection_id]
    logger.info(f"❌ Client disconnected: {connection_id}")
    return {"statusCode": 200}

def handle_init_conversation(event):
    """Initialize conversation with orchestrator"""
    try:
        connection_id = event["requestContext"]["connectionId"]
        
        # Create orchestrator for this connection
        orchestrators[connection_id] = TaskOrchestrator(connection_id)
        chat_histories[connection_id] = []
        
        # Get initial action from orchestrator
        action = orchestrators[connection_id].get_next_action()
        
        return execute_action(connection_id, action, event)
        
    except Exception as e:
        logger.error(f"❗ initConversation failed: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

def handle_send_message(event):
    """Main message handler - routes through orchestrator"""
    try:
        connection_id = event["requestContext"]["connectionId"]
        body = json.loads(event.get("body", "{}"))
        user_msg = body.get("message", "")
        
        # Check if in interactive analytics mode (legacy support)
        interactive_state = get_interactive_state(connection_id)
        if interactive_state and interactive_state["mode"] == "interactive_analytics":
            logger.info(f"🕵️ User in interactive analytics mode")
            return handle_interactive_analytics_response(connection_id, user_msg, interactive_state["conversation_state"], event)
        
        # Get or create orchestrator
        if connection_id not in orchestrators:
            orchestrators[connection_id] = TaskOrchestrator(connection_id)
            chat_histories[connection_id] = []
        
        orchestrator = orchestrators[connection_id]
        
        # Add user message to history
        if user_msg and user_msg.strip():
            chat_histories[connection_id].append({"role": "user", "content": user_msg})
        
        # Handle special messages
        if user_msg.strip() == "Processing complete. The map data is ready for LLM":
            # If we're in parallel info gathering, move to detailed investigation
            # If we're in location processing, move to basic info gathering
            if orchestrator.state.phase == ConversationPhase.PARALLEL_INFO_GATHERING:
                orchestrator.state.phase = ConversationPhase.ANALYTICS_PROCESSING
            else:
                orchestrator.state.phase = ConversationPhase.BASIC_INFO_GATHERING
        
        # Get next action from orchestrator
        action = orchestrator.get_next_action(user_msg)
        
        # Execute the action
        return execute_action(connection_id, action, event)
        
    except Exception as e:
        logger.error(f"❗ Unhandled error in send_message: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}

# --- Action Executors ---
def execute_action(connection_id: str, action: Dict[str, Any], event) -> Dict[str, int]:
    """Execute actions determined by orchestrator"""
    try:
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        action_type = action.get("action")
        
        if action_type == "claude_response":
            return handle_claude_response(connection_id, action, apig)
            
        elif action_type == "start_parallel_processing":
            return handle_start_parallel_processing(connection_id, action, apig, event)
            
        elif action_type == "check_map_processing":
            return handle_check_map_processing(connection_id, action, apig, event)
            
        elif action_type == "acknowledge_location":
            return handle_acknowledge_location(connection_id, action, apig)
            
        elif action_type == "trigger_analytics":
            return handle_trigger_analytics(connection_id, action, apig, event)
            
        elif action_type == "start_detailed_analytics":
            return handle_start_detailed_analytics(connection_id, action, apig, event)
            
        elif action_type == "interactive_investigation":
            return handle_interactive_investigation(connection_id, action, apig, event)
            
        elif action_type == "run_similarity_search":
            return handle_similarity_search(connection_id, action, apig)
            
        elif action_type == "finalize_case":
            return handle_finalize_case(connection_id,action, apig)
            
        else:
            logger.error(f"Unknown action type: {action_type}")
            return {"statusCode": 400}
            
    except Exception as e:
        logger.error(f"❗ Error executing action: {e}")
        return {"statusCode": 500}


def handle_start_parallel_processing(connection_id: str, action: Dict[str, Any], apig, event) -> Dict[str, int]:
    """NEW: Handle parallel processing start - trigger map analysis and ask first question"""
    try:
        message = action.get("message")
        progress = action.get("progress", "")
        
        # Send the acknowledgment message
        chat_histories[connection_id].append({"role": "assistant", "content": message})
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": message,
                "connectionId": connection_id,
                "progress": progress,
                "type": "parallel_processing_started"
            }).encode("utf-8")
        )
        
        # TODO: Here you would trigger the map processing in the background
        
        # Immediately ask the first question
        orchestrator = orchestrators[connection_id]
        next_action = orchestrator.get_next_action()
        
        # Execute the next action (which should ask the first question)
        if next_action:
            return execute_action(connection_id, next_action, event)
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Parallel processing start error: {e}")
        return {"statusCode": 500}

def handle_check_map_processing(connection_id: str, action: Dict[str, Any], apig, event) -> Dict[str, int]:
    """NEW: Check if map processing is complete and proceed accordingly"""
    try:
        # Check if map processing is done (you can implement actual checking logic here)
        # For now, we'll assume it's ready and move to detailed analytics
        
        message = action.get("message", "Location analysis is complete! Now I'll ask some specific questions based on the road layout.")
        progress = action.get("progress", "")
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": message,
                "connectionId": connection_id,
                "progress": progress
            }).encode("utf-8")
        )
        
        # Move to analytics processing
        orchestrators[connection_id].state.phase = ConversationPhase.ANALYTICS_PROCESSING
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Map processing check error: {e}")
        return {"statusCode": 500}

def handle_claude_response(connection_id: str, action: Dict[str, Any], apig) -> Dict[str, int]:
    """Handle Claude responses with orchestrator instructions - FIXED to include JST time context"""
    try:
        instruction = action.get("instruction", "Continue the conversation naturally.")
        progress = action.get("progress", "")
        
        # FIXED: Add JST time context
        jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))
        time_context = f"FYI, today is \"{jst_time.strftime('%Y/%m/%d %H:%M')} JST\". "
        
        # Add instruction with time context to chat history
        full_instruction = time_context + instruction
        chat_histories[connection_id].append({"role": "user", "content": full_instruction})
        
        # Get Claude response
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
        
        # Send response with progress
        message_data = {
            "response": completion,
            "connectionId": connection_id,
            "progress": progress
        }
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(message_data).encode("utf-8")
        )
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Claude response error: {e}")
        return {"statusCode": 500}

def handle_acknowledge_location(connection_id: str, action: Dict[str, Any], apig) -> Dict[str, int]:
    """Handle location acknowledgment"""
    try:
        message = action.get("message")
        progress = action.get("progress", "")
        
        chat_histories[connection_id].append({"role": "assistant", "content": message})
        
        message_data = {
            "response": message,
            "connectionId": connection_id,
            "progress": progress,
            "type": "location_acknowledged"
        }
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(message_data).encode("utf-8")
        )
        
        # Move to basic info gathering immediately
        orchestrators[connection_id].state.phase = ConversationPhase.BASIC_INFO_GATHERING
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Location acknowledgment error: {e}")
        return {"statusCode": 500}

def handle_trigger_analytics(connection_id: str, action: Dict[str, Any], apig, event) -> Dict[str, int]:
    """Trigger initial analytics processing"""
    try:
        # This is where "Processing complete. The map data is ready for LLM" is handled
        orchestrators[connection_id].state.phase = ConversationPhase.BASIC_INFO_GATHERING
        
        message = "Location analysis complete. Now let me gather some basic information about your accident."
        progress = action.get("progress", "")
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": message,
                "connectionId": connection_id,
                "progress": progress
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Analytics trigger error: {e}")
        return {"statusCode": 500}

def handle_start_detailed_analytics(connection_id: str, action: Dict[str, Any], apig, event) -> Dict[str, int]:
    """Start detailed analytics with interactive investigation - FIXED data mapping"""
    try:
        # Get orchestrator data
        orchestrator = orchestrators[connection_id]
        orchestrator_data = orchestrator.state.collected_data
        
        # FIXED: Comprehensive data mapping with debugging
        collected_data = {}
        
        # Debug: Log what we have
        logger.info(f"🔍 Orchestrator collected data: {orchestrator_data}")
        logger.info(f"🔍 Questions asked status: {orchestrator.state.questions_asked}")
        
        # Priority 1: Direct keys from parallel/basic gathering
        datetime_value = None
        description_value = None
        
        # Check all possible datetime keys
        for key in ["datetime", "parallel_q_0", "basic_q_0"]:
            if key in orchestrator_data and orchestrator_data[key]:
                datetime_value = orchestrator_data[key]
                logger.info(f"✅ Found datetime in key: {key} = {datetime_value}")
                break
        
        # Check all possible description keys  
        for key in ["description", "parallel_q_1", "basic_q_1"]:
            if key in orchestrator_data and orchestrator_data[key]:
                description_value = orchestrator_data[key]
                logger.info(f"✅ Found description in key: {key} = {description_value}")
                break
        
        # Map to analytics expected format
        if datetime_value:
            collected_data["basic_q_0"] = datetime_value
        if description_value:
            collected_data["basic_q_1"] = description_value
            
        logger.info(f"📤 Sending to analytics: {collected_data}")
        
        # Call analytics function
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'initial_analysis',
                'collected_data': collected_data
            })
        )
        
        result = json.loads(response['Payload'].read())
        
        if result.get("statusCode") == 200:
            # Store analytics result
            orchestrators[connection_id].state.analytics_result = result
            
            # Move to detailed investigation
            orchestrators[connection_id].state.phase = ConversationPhase.DETAILED_INVESTIGATION
            
            first_question = result.get('message', 'Based on the location analysis, let me ask you some detailed questions about the accident.')
            
            # Store interactive state
            store_interactive_state(connection_id, result.get('conversation_state', 'awaiting_direction'))
            
            apig.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "response": first_question,
                    "connectionId": connection_id,
                    "progress": "Detailed investigation (Step 5/6)"
                }).encode("utf-8")
            )
            
            return {"statusCode": 200}
        else:
            # Fallback to similarity search
            orchestrators[connection_id].state.phase = ConversationPhase.SIMILARITY_ANALYSIS
            return handle_similarity_search(connection_id, {"action": "run_similarity_search"}, apig)
            
    except Exception as e:
        logger.error(f"❗ Detailed analytics error: {e}")
        return {"statusCode": 500}


def handle_interactive_investigation(connection_id: str, action: Dict[str, Any], apig, event) -> Dict[str, int]:
    """Handle interactive investigation responses"""
    try:
        user_input = action.get("user_input", "")
        
        # Call analytics for follow-up
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'user_response',
                'user_input': user_input,
                'conversation_state': orchestrators[connection_id].state.analytics_result.get('conversation_state')
            })
        )
        
        result = json.loads(response['Payload'].read())
        
        if result.get("statusCode") == 200:
            message = result.get("message", "Thank you for that information.")
            
            # Update interactive state
            new_state = result.get("conversation_state")
            if new_state:
                store_interactive_state(connection_id, new_state)
            
            # Check if investigation is complete
            if result.get("animation_ready") or result.get("investigation_complete"):
                orchestrators[connection_id].state.phase = ConversationPhase.SIMILARITY_ANALYSIS
                message += "\n\nExcellent! Now let me find similar accident cases for comparison..."
                
                # Clear interactive state
                clear_interactive_state(connection_id)
                
                apig.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({
                        "response": message,
                        "connectionId": connection_id,
                        "progress": "Investigation complete (Step 5/6)"
                    }).encode("utf-8")
                )
                
                # Trigger similarity search
                return handle_similarity_search(connection_id, {"action": "run_similarity_search"}, apig)
            else:
                # Continue investigation
                apig.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({
                        "response": message,
                        "connectionId": connection_id,
                        "progress": action.get("progress", "Detailed investigation (Step 5/6)")
                    }).encode("utf-8")
                )
                
            return {"statusCode": 200}
        else:
            # Move to similarity search if analytics fails
            orchestrators[connection_id].state.phase = ConversationPhase.SIMILARITY_ANALYSIS
            return handle_similarity_search(connection_id, {"action": "run_similarity_search"}, apig)
            
    except Exception as e:
        logger.error(f"❗ Interactive investigation error: {e}")
        return {"statusCode": 500}


def handle_similarity_search(connection_id: str, action: Dict[str, Any], apig) -> Dict[str, int]:
    """Handle similarity search and case matching"""
    try:
        # Get full conversation context
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[connection_id]])
        
        # Check if we have enough info for similarity search
        if should_run_similarity_search(full_context):
            apig.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "response": "Analyzing your case and comparing with similar accidents in our database...",
                    "connectionId": connection_id,
                    "progress": "Finding similar cases (Step 6/6)"
                }).encode("utf-8")
            )
            
            # Run similarity search
            similar_cases = find_similar_cases(full_context)
            
            if similar_cases:
                # Generate follow-up questions based on modifiers
                modifiers = set()
                for case in similar_cases:
                    modifiers.update(case.get("modifications", []))
                modifiers = list(modifiers)[:5]
                
                if modifiers:
                    mod_questions = generate_modifier_questions(modifiers)
                    
                    # Store modifier questions in orchestrator state
                    orchestrators[connection_id].state.modifier_questions = mod_questions
                    orchestrators[connection_id].state.current_modifier_index = 0
                    
                    # Move to legal modifier investigation phase
                    orchestrators[connection_id].state.phase = ConversationPhase.LEGAL_MODIFIER_INVESTIGATION
                    
                    intro_msg = (
                        "I found some similar cases in our database. To provide the most accurate fault ratio assessment, "
                        "I need to ask a few specific questions about the circumstances. These help determine legal liability factors."
                    )
                    
                    apig.post_to_connection(
                        ConnectionId=connection_id,
                        Data=json.dumps({
                            "response": intro_msg,
                            "connectionId": connection_id,
                            "progress": "Preparing legal assessment (Step 6/6)"
                        }).encode("utf-8")
                    )
                    
                    return {"statusCode": 200}
        
        # Move to finalization if no modifiers or not enough context
        orchestrators[connection_id].state.phase = ConversationPhase.CASE_FINALIZATION
        return handle_finalize_case(connection_id, {"action": "finalize_case"}, apig)
        
    except Exception as e:
        logger.error(f"❗ Similarity search error: {e}")
        return {"statusCode": 500}



def handle_finalize_case(connection_id: str, action: Dict[str, Any], apig) -> Dict[str, int]:
    """Finalize the case"""
    try:
        # Extract final information
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_histories[connection_id]])
        datetime_str = extract_datetime(full_context)
        location_data = extract_location(full_context)
        
        # Store to DynamoDB
        if datetime_str != "unknown" and location_data:
            similar_cases = find_similar_cases(full_context)
            store_to_dynamodb(connection_id, datetime_str, location_data, similar_cases)
        
        final_message = (
            "Thank you for providing all the necessary information about your accident. "
            "I have recorded all the details and your claim is now being processed. "
            "You should receive further updates within 24-48 hours. "
            "Is there anything else you'd like to clarify about the accident?"
        )
        
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": final_message,
                "connectionId": connection_id,
                "progress": "Claim processing complete ✅",
                "status": "completed"
            }).encode("utf-8")
        )
        
        orchestrators[connection_id].state.phase = ConversationPhase.COMPLETED
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Case finalization error: {e}")
        return {"statusCode": 500}

# --- Interactive Analytics Support (Legacy) ---
def handle_interactive_analytics_response(connection_id, user_input, current_state, event):
    """Handle user responses during interactive investigation (legacy support) - IMPROVED error handling"""
    try:
        logger.info(f"🤖 Processing analytics response: '{user_input}'")
        
        # Call analytics function
        response = lambda_client.invoke(
            FunctionName=ANALYTICS_FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'connection_id': connection_id,
                'bucket_name': BUCKET,
                'message_type': 'user_response',
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
                
                # Move orchestrator to similarity analysis
                if connection_id in orchestrators:
                    orchestrators[connection_id].state.phase = ConversationPhase.SIMILARITY_ANALYSIS
            
            apig.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({
                    "response": message,
                    "connectionId": connection_id
                }).encode("utf-8")
            )
            
            return {"statusCode": 200}
        else:
            # Analytics function returned an error
            error_msg = result.get("errorMessage", "Unknown error")
            logger.error(f"❗ Analytics function returned error for {connection_id}: {error_msg}")
            logger.error(f"❗ Full analytics result: {result}")
            raise Exception(f"Analytics function error: {error_msg}")
            
    except json.JSONDecodeError as e:
        logger.error(f"❗ Failed to parse analytics response for {connection_id}: {e}")
        logger.error(f"❗ Raw response: {response.get('Payload', 'No payload')}")
        clear_interactive_state(connection_id)
        
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "I'm having trouble processing the detailed analysis. Let me analyze your case and compare with similar accidents in our database...",
                "connectionId": connection_id
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}
        
    except Exception as e:
        logger.error(f"❗ Interactive analytics failed for {connection_id}: {e}")
        logger.error(f"❗ Current state: {current_state}")
        logger.error(f"❗ User input: {user_input}")
        logger.error(f"❗ Analytics function name: {ANALYTICS_FUNCTION_NAME}")
        logger.error(f"❗ Bucket: {BUCKET}")
        
        # Log the full stack trace
        import traceback
        logger.error(f"❗ Full traceback: {traceback.format_exc()}")
        
        clear_interactive_state(connection_id)
        
        apig = get_apig_client(event["requestContext"]["domainName"], event["requestContext"]["stage"])
        apig.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({
                "response": "I encountered an issue with the detailed investigation. Let me analyze your case and compare with similar accidents in our database...",
                "connectionId": connection_id
            }).encode("utf-8")
        )
        
        return {"statusCode": 200}

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

def clear_interactive_state(connection_id):
    """Clear interactive analytics state"""
    try:
        state_key = f"interactive-state/{connection_id}/state.json"
        s3.delete_object(Bucket=BUCKET, Key=state_key)
        logger.info(f"🧹 Cleared interactive state for {connection_id}")
    except Exception as e:
        logger.error(f"❗ Failed to clear interactive state: {e}")

# --- Helper Functions ---
def generate_modifier_questions(modifiers: list[str]) -> list[str]:
    # FIXED: Add JST time context here too
    jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))
    time_context = f"FYI, today is \"{jst_time.strftime('%Y/%m/%d %H:%M')} JST\". "
    
    prompt = (
        time_context +
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
    # FIXED: Add JST time context
    jst_time = datetime.now(ZoneInfo("Asia/Tokyo"))
    time_context = f"FYI, today is \"{jst_time.strftime('%Y/%m/%d %H:%M')} JST\". "
    
    prompt = time_context + f"Generate the exact datetime (format: yyyy/mm/dd hh:mm) from this accident description: \"{full_context}\". If the user provides a relative date expression such as yesterday, today, or three days ago, instead of a specific date, estimate the exact date by referring to today's date. Just output yyyy/mm/dd hh:mm directly and Don't include any other messages. If you can't do it, return 'unknown'."
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

    json_match = re.search(r"\\$\\$\s*\{.*?\}\s*\\$\\$", result_text, re.DOTALL)
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
    matches = find_matching_pattern(category, full_context)

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

def store_to_dynamodb(connection_id, datetime_str, location, similar_cases=None):
    try:
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
            } for case in similar_cases[:5]]

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
            
    except Exception as e:
        logger.error(f"❗ Failed to store to DynamoDB: {e}")

# --- REST API Handlers ---
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
    logger.info(f"🧭 RouteKey: {route_key}")
    
    if route_key:
        if route_key == "$connect":
            return handle_connect(event)
        elif route_key == "$disconnect":
            return handle_disconnect(event)
        elif route_key == "sendMessage":
            return handle_send_message(event)
        elif route_key == "initConversation":
            return handle_init_conversation(event)
        elif route_key == "$default":
            body = _parse_body(event)
            if body.get("action") == "ping":
                return {"statusCode": 200}
            logger.warning(f"Unknown action via default: {body}")
            return {"statusCode": 200}
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