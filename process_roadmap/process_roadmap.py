import os
import boto3
import json
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ecs = boto3.client("ecs")
s3 = boto3.client("s3")

def lambda_handler(event, context):
    base_name = event.get("base_name")  # 例: "00032_35.67326_139.48047"
    
    if not base_name:
        logger.error("❗ No base_name provided in event")
        return {
            'statusCode': 400,
            'body': "Missing base_name parameter"
        }
    
    cluster_name = os.environ['CLUSTER_NAME']
    service_name = os.environ.get('FIELD_SERVICE_NAME')  # NEW: Service name
    task_def = os.environ['TASK_DEF']
    
    logger.info(f"🚀 Processing roadmap for: {base_name}")
    
    # Try to use hot standby service first
    if service_name and try_service_field_generation(cluster_name, service_name, base_name):
        logger.info(f"✅ Used hot standby service: {service_name}")
        return {
            'statusCode': 200,
            'body': f"Queued work for service: {service_name}"
        }
    
    # Fallback to original run_task method
    logger.info("⏱️ Falling back to run_task method")
    response = run_task_field_generation(cluster_name, task_def, base_name)
    
    return {
        'statusCode': 200,
        'body': f"Started Fargate task: {response['tasks'][0]['taskArn']}"
    }

def try_service_field_generation(cluster_name, service_name, base_name):
    """Try to use running service containers"""
    try:
        # Check if service has running containers
        if not check_service_availability(cluster_name, service_name):
            logger.warning(f"⚠️ Service {service_name} not available")
            return False
        
        # Queue work for the service
        work_data = {
            "base_name": base_name,
            "command": [base_name],
            "work_type": "field_generation"
        }
        
        success = queue_work_for_service(base_name, work_data)
        
        if success:
            # Optionally scale up service to handle the work
            scale_service_if_needed(cluster_name, service_name, min_desired=1)
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"❗ Service field generation attempt failed: {e}")
        return False

def run_task_field_generation(cluster_name, task_def, base_name):
    """Original run_task method as fallback"""
    return ecs.run_task(
        cluster=cluster_name,
        launchType='FARGATE',
        taskDefinition=task_def,
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [os.environ['SUBNET_ID']],
                'securityGroups': [os.environ['SECURITY_GROUP']],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'field-generator',
                    'command': [base_name],
                    'environment': [
                        {'name': 'RUN_MODE', 'value': 'TASK'}  # NEW: Force task mode
                    ]
                }
            ]
        }
    )

def check_service_availability(cluster_name, service_name):
    """Check if ECS service has running containers"""
    try:
        response = ecs.describe_services(
            cluster=cluster_name,
            services=[service_name]
        )
        
        if not response['services']:
            logger.warning(f"⚠️ Service {service_name} not found")
            return False
            
        service = response['services'][0]
        running_count = service['runningCount']
        
        logger.info(f"🔍 Service {service_name}: {running_count} running containers")
        return running_count > 0
        
    except Exception as e:
        logger.error(f"❗ Error checking service availability: {e}")
        return False

def queue_work_for_service(base_name, work_data):
    """Queue work item for service containers to pick up"""
    try:
        bucket_name = os.environ.get('FIELD_OUTPUT_BUCKET')  # You'll need to add this env var
        logger.info(f"🔍 FIELD_OUTPUT_BUCKET from env: {os.environ.get('FIELD_OUTPUT_BUCKET')}")
        if not bucket_name:
            logger.error("❗ FIELD_OUTPUT_BUCKET environment variable not set")
            return False
            
        work_item = {
            "base_name": base_name,
            "work_data": work_data,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Put work item in S3 queue location
        work_key = f"work-queue/field_generation/{base_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=work_key,
            Body=json.dumps(work_item, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"📝 Queued field generation work: {work_key}")
        return True
        
    except Exception as e:
        logger.error(f"❗ Failed to queue work: {e}")
        return False

def scale_service_if_needed(cluster_name, service_name, min_desired=1):
    """Scale up service if it has no running containers"""
    try:
        response = ecs.describe_services(
            cluster=cluster_name,
            services=[service_name]
        )
        
        if not response['services']:
            logger.warning(f"⚠️ Service {service_name} not found for scaling")
            return False
            
        service = response['services'][0]
        current_desired = service['desiredCount']
        current_running = service['runningCount']
        
        if current_running  < min_desired:
            # Scale up the service
            new_desired = max(current_desired + 1, min_desired)
            
            ecs.update_service(
                cluster=cluster_name,
                service=service_name,
                desiredCount=new_desired
            )
            
            logger.info(f"🔥 Scaled service {service_name}: {current_desired} → {new_desired}")
            return True
        
        return True  # Already has enough containers
        
    except Exception as e:
        logger.error(f"❗ Failed to scale service: {e}")
        return False