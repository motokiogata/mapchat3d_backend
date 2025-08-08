import json
import boto3
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

def lambda_handler(event, context):
    """
    Analyzes S3 JSON data using Bedrock and returns traffic infrastructure insights
    """
    try:
        connection_id = event['connection_id']
        bucket_name = event['bucket_name']
        chat_history = event.get('chat_history', [])
        
        logger.info(f"🔍 Starting analytics for connection: {connection_id}")
        
        # Define the expected files and their purposes
        expected_files = {
            'centerlines_with_metadata.json': 'road centerlines and traffic control data',
            'integrated_road_network.json': 'complete road network structure',
            'intersections_with_metadata.json': 'intersection details and traffic signals',
            'lane_tree_routes_enhanced.json': 'lane-level routing and navigation data'
        }
        
        # Get conversation context to understand customer's route
        customer_context = extract_customer_route_context(chat_history)
        
        # Analyze each file
        file_analyses = {}
        s3_prefix = f"outputs/{connection_id}/"
        
        for filename, description in expected_files.items():
            try:
                key = f"{s3_prefix}{filename}"
                logger.info(f"📄 Analyzing file: {key}")
                
                # Download and parse JSON
                file_response = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = file_response['Body'].read().decode('utf-8')
                data = json.loads(file_content)
                
                # Use Bedrock to analyze the specific file
                analysis = analyze_traffic_data_with_bedrock(
                    data, filename, description, customer_context
                )
                file_analyses[filename] = analysis
                
            except Exception as e:
                logger.warning(f"⚠️ Could not analyze {filename}: {e}")
                file_analyses[filename] = f"File not available or error: {str(e)}"
        
        # Generate comprehensive traffic infrastructure report
        final_report = generate_traffic_infrastructure_report(
            file_analyses, customer_context, chat_history
        )
        
        logger.info(f"✅ Traffic analytics completed for connection: {connection_id}")
        return {
            "statusCode": 200,
            "analytics_report": final_report
        }
        
    except Exception as e:
        logger.error(f"❗ Analytics function error: {e}")
        return {
            "statusCode": 500,
            "analytics_report": f"Error analyzing traffic data: {str(e)}"
        }

def extract_customer_route_context(chat_history: List[Dict]) -> str:
    """
    Extract customer's route information from chat history
    """
    full_context = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in chat_history 
        if msg['role'] in ['user', 'assistant']
    ])
    
    prompt = f"""
    From this conversation, extract information about the customer's vehicle route and movement:
    
    {full_context}
    
    Please identify and summarize:
    1. Where the customer was coming from (origin direction)
    2. Where the customer was going (destination direction)  
    3. What maneuver the customer was making (straight, left turn, right turn, etc.)
    4. The customer's lane position
    5. Any mentioned traffic controls (signals, signs, etc.)
    
    If information is not available, state "Not specified".
    Format as clear bullet points.
    """
    
    try:
        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        route_context = json.loads(response["body"].read())["content"][0]["text"]
        return route_context.strip()
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract route context: {e}")
        return "Customer route information not available"

def analyze_traffic_data_with_bedrock(data: Dict, filename: str, description: str, customer_context: str) -> str:
    """
    Use Bedrock to analyze specific traffic infrastructure data
    """
    # Convert data to string for analysis (truncate if too large)
    data_str = json.dumps(data, indent=2)
    if len(data_str) > 15000:  # Truncate large data
        data_str = data_str[:15000] + "\n... (truncated for analysis)"
    
    if filename == 'intersections_with_metadata.json':
        prompt = f"""
        Analyze this intersection data to identify traffic infrastructure:
        
        Customer Context: {customer_context}
        
        Data: {data_str}
        
        Please identify:
        1. Are there traffic signals at this intersection? (Yes/No and details)
        2. Are there stop signs? (Yes/No and locations)
        3. What type of intersection is this? (4-way, T-intersection, roundabout, etc.)
        4. Are there crosswalks present?
        5. What are the road names at this intersection?
        6. Based on the customer's route, which direction were they likely traveling?
        
        Provide clear, factual analysis.
        """
        
    elif filename == 'centerlines_with_metadata.json':
        prompt = f"""
        Analyze this road centerline data for traffic control features:
        
        Customer Context: {customer_context}
        
        Data: {data_str}
        
        Please identify:
        1. Are there stop lines marked on the road?
        2. What lane markings are present? (solid, dashed, turn lanes, etc.)
        3. Are there dedicated turn lanes?
        4. What is the road geometry? (straight, curved, etc.)
        5. Based on customer context, which road/lane was the customer likely using?
        
        Focus on infrastructure that affects traffic flow and right-of-way.
        """
        
    elif filename == 'lane_tree_routes_enhanced.json':
        prompt = f"""
        Analyze this lane routing data:
        
        Customer Context: {customer_context}
        
        Data: {data_str}
        
        Please identify:
        1. What are the available lane configurations?
        2. Which lanes allow which movements? (straight, left turn, right turn)
        3. Are there any lane restrictions or special rules?
        4. Based on the customer's intended movement, which lane should they have been in?
        5. Are there any merge points or lane changes required?
        
        Connect this to the customer's described route.
        """
        
    else:  # integrated_road_network.json
        prompt = f"""
        Analyze this integrated road network data:
        
        Customer Context: {customer_context}
        
        Data: {data_str}
        
        Please identify:
        1. What is the overall road network structure?
        2. How many roads/streets intersect at this location?
        3. What are the traffic flow patterns?
        4. Are there any special traffic control devices or rules?
        5. How does this network relate to the customer's described path?
        
        Provide a comprehensive infrastructure overview.
        """
    
    try:
        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        analysis = json.loads(response["body"].read())["content"][0]["text"]
        return analysis.strip()
        
    except Exception as e:
        logger.error(f"❗ Bedrock analysis failed for {filename}: {e}")
        return f"Analysis failed: {str(e)}"

def generate_traffic_infrastructure_report(file_analyses: Dict[str, str], customer_context: str, chat_history: List[Dict]) -> str:
    """
    Generate a comprehensive traffic infrastructure report using Bedrock
    """
    # Combine all analyses
    combined_analyses = "\n\n".join([
        f"=== {filename.upper()} ANALYSIS ===\n{analysis}"
        for filename, analysis in file_analyses.items()
    ])
    
    prompt = f"""
    You are a traffic accident analyst reviewing infrastructure data for an insurance claim.
    
    CUSTOMER ROUTE CONTEXT:
    {customer_context}
    
    DETAILED INFRASTRUCTURE ANALYSES:
    {combined_analyses}
    
    Please provide a comprehensive summary that answers these key questions:
    
    1. **Traffic Signals**: Are there traffic signals at this intersection? What was their likely status?
    
    2. **Stop Signs/Lines**: Are there stop signs or stop lines? Where are they located?
    
    3. **Intersection Type**: What type of intersection is this and how does it affect right-of-way?
    
    4. **Road Layout**: Describe the road configuration and lane structure
    
    5. **Customer's Path**: Based on the infrastructure and customer's description, analyze:
       - Which lane was the customer likely in?
       - What traffic controls would have applied to them?
       - Were they following the correct route for their intended movement?
    
    6. **Right-of-Way Analysis**: Who should have had right-of-way based on the infrastructure?
    
    7. **Key Infrastructure Factors**: What infrastructure elements are most relevant to accident liability?
    
    Format your response as a clear, professional analysis that Mariko (the insurance operator) can use to explain the scene to the customer.
    """
    
    try:
        response = bedrock.invoke_model(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1200,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        final_report = json.loads(response["body"].read())["content"][0]["text"]
        return final_report.strip()
        
    except Exception as e:
        logger.error(f"❗ Failed to generate final report: {e}")
        return f"Error generating traffic infrastructure report: {str(e)}"