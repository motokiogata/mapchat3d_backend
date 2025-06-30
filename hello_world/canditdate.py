import json
import boto3
import os
import logging
import pandas as pd
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

# Add these new components to your existing Lambda
s3 = boto3.client('s3')
CSV_BUCKET = os.environ.get('CSV_BUCKET_NAME', 'your-csv-bucket')

# Category mapping - adjust based on your 7 categories and 12 CSV files
CATEGORY_CSV_MAPPING = {
    "Vehicle(car_or_motorcycle)_accident_against_pedestrian": ["1-25.csv", "26-50.csv"],
    "Bycicle_accident_against_pedestrian": ["51-74.csv", "75-97.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_motorcycle)": ["98-113.csv", "114-138.csv", "139-159.csv"],
    "Vehicle_to_vehicle_(car_accidents_against_car)": ["160-204.csv"," 205-234.csv"],
    "Vehicle(car_or_motorcycle)_accident_against_bycicle": ["235-280.csv", "281-310.csv"],
    "Accidents_in_highways_or_Accidents_in_park": ["311-338.csv"]
}

def categorize_accident_type(full_context):
    """Step 1: Determine main accident category"""
    prompt = f"""
    Analyze this accident description and categorize it into ONE of these categories:
    1. Vehicle(car_or_motorcycle)_accident_against_pedestrian
    2. Bycicle_accident_against_pedestrian
    3. Vehicle_to_vehicle_(car_accidents_against_motorcycle)
    4. Vehicle_to_vehicle_(car_accidents_against_car)
    5. Vehicle(car_or_motorcycle)_accident_against_bycicle
    6. Accidents_in_highways_or_Accidents_in_park
    
    Conversation: {full_context}
    
    Respond with ONLY the category name (like Vehicle(car_or_motorcycle)_accident_against_pedestrian). No explanation and No number.
    """
    
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 30,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        category = json.loads(response["body"].read())["content"][0]["text"].strip()
        logger.info(f"🏷️ Categorized as: {category}")
        return category if category in CATEGORY_CSV_MAPPING else "other_accidents"
        
    except Exception as e:
        logger.error(f"Error in categorization: {e}")
        return "other_accidents"

def load_csv_from_s3(csv_filename):
    """Load CSV file from S3"""
    try:
        response = s3.get_object(Bucket=CSV_BUCKET, Key=csv_filename)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"📄 Loaded {csv_filename}: {len(df)} cases")
        return df
    except Exception as e:
        logger.error(f"Error loading {csv_filename}: {e}")
        return None

def find_matching_pattern(category, full_context):
    """Step 2: Load relevant CSVs and find matching pattern"""
    csv_files = CATEGORY_CSV_MAPPING.get(category, [])
    if not csv_files:
        logger.warning(f"No CSV files found for category: {category}")
        return None
    
    # Load all relevant CSV files
    all_cases = []
    for csv_file in csv_files:
        df = load_csv_from_s3(csv_file)
        if df is not None:
            for _, row in df.iterrows():
                all_cases.append({
                    "case_number": row["Case Number"],
                    "situation": row["Accident Situation"],
                    "fault_ratio": row["Fault Ratio"],
                    "modification_factors": row["Modification Factors"],
                    "source_file": csv_file
                })
    
    if not all_cases:
        logger.warning(f"No cases loaded for category: {category}")
        return None
    
    # Format cases for LLM analysis
    cases_text = format_cases_for_analysis(all_cases)
    
    # Find best matching pattern
    matching_prompt = f"""
    Based on this accident description, find the MOST SIMILAR case from the following patterns:

    ACCIDENT DESCRIPTION:
    {full_context}

    AVAILABLE PATTERNS:
    {cases_text}

    Analyze each pattern and return the result in this exact JSON format:
    {{
        "best_match_case_number": "[X]",
        "confidence_score": 8.5,
        "fault_ratio": "XX%",
        "applicable_modifications": ["factor1", "factor2"],
        "reasoning": "Brief explanation of why this case matches"
    }}

    Only return the JSON, no other text.
    """
    
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": matching_prompt}]
            })
        )
        
        result_text = json.loads(response["body"].read())["content"][0]["text"].strip()
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            pattern_result = json.loads(json_match.group(0))
            logger.info(f"🎯 Found matching pattern: {pattern_result}")
            return pattern_result
        
    except Exception as e:
        logger.error(f"Error finding matching pattern: {e}")
    
    return None

def format_cases_for_analysis(cases):
    """Format cases for LLM analysis"""
    formatted = ""
    for case in cases[:20]:  # Limit to prevent token overflow
        formatted += f"""
Case {case['case_number']}:
Situation: {case['situation']}
Base Fault Ratio: {case['fault_ratio']}
Modification Factors: {case['modification_factors']}
---
"""
    return formatted

def calculate_final_fault_ratio(base_ratio, modifications, accident_context):
    """Calculate final fault ratio with modifications"""
    calculation_prompt = f"""
    Calculate the final fault ratio based on:
    
    Base Fault Ratio: {base_ratio}
    Available Modifications: {modifications}
    Accident Context: {accident_context}
    
    Determine which modifications apply and calculate the final percentage.
    
    Return only the final percentage number (e.g., "45%").
    """
    
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json", 
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": calculation_prompt}]
            })
        )
        
        result = json.loads(response["body"].read())["content"][0]["text"].strip()
        logger.info(f"📊 Calculated final fault ratio: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating fault ratio: {e}")
        return base_ratio

# Update your existing find_similar_case function
def find_similar_case(full_context):
    """Enhanced version with dynamic CSV loading"""
    try:
        # Step 1: Categorize the accident
        category = categorize_accident_type(full_context)
        logger.info(f"🔍 Starting analysis for category: {category}")
        
        # Step 2: Find matching pattern in relevant CSVs
        pattern_match = find_matching_pattern(category, full_context)
        
        if pattern_match:
            # Step 3: Calculate final fault ratio
            final_ratio = calculate_final_fault_ratio(
                pattern_match["fault_ratio"],
                pattern_match.get("applicable_modifications", []),
                full_context
            )
            
            # Return structured result
            return [{
                "case_id": pattern_match["best_match_case_number"],
                "category": category,
                "similarity": float(pattern_match["confidence_score"]) / 10.0,
                "summary": pattern_match["reasoning"],
                "fault_ratio": final_ratio,
                "base_fault_ratio": pattern_match["fault_ratio"],
                "modifications": pattern_match.get("applicable_modifications", [])
            }]
        
        return None
        
    except Exception as e:
        logger.error(f"Error in enhanced similarity search: {e}")
        return None

# Update your DynamoDB storage to include new fields
def store_to_dynamodb(connection_id, datetime_str, location, similar_cases=None):
    response = table.get_item(Key={"connection_id": connection_id})
    item = response.get("Item")

    similar_cases_data = None
    if similar_cases and len(similar_cases) > 0:
        similar_cases_data = []
        for case in similar_cases[:3]:
            case_data = {
                "case_id": str(case["case_id"]),
                "category": case.get("category", "unknown"),
                "similarity": Decimal(str(case["similarity"])),
                "summary": case["summary"][:500],
                "fault_ratio": case.get("fault_ratio", "unknown"),
                "base_fault_ratio": case.get("base_fault_ratio", "unknown")
            }
            
            if case.get("modifications"):
                case_data["modifications"] = case["modifications"][:5]  # Limit array size
                
            similar_cases_data.append(case_data)

    # Rest of your existing DynamoDB logic...
    if item:
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
        new_item = {
            "connection_id": connection_id,
            "timestamp": datetime_str,
            "lat": Decimal(str(location["lat"])),
            "lon": Decimal(str(location["lon"]))
        }
        
        if similar_cases_data:
            new_item["similar_cases"] = similar_cases_data
            
        table.put_item(Item=new_item)