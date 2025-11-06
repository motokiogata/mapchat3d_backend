#docker-entrypoint.py of docker_fargate is like this now.

import os
import sys
import boto3
import time
import json
import signal
import logging
from download_from_s3_with_retry import download_with_retry
import subprocess
from s3_handler import S3Handler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client("s3")

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info('📡 Shutdown signal received, finishing current work...')
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def upload_done_flag(connection_id, bucket):
    print(f"🔍 Attempting to upload to bucket: {bucket}")
    print(f"🔍 Key will be: outputs/{connection_id}/done.flag")
    
    key = f"outputs/{connection_id}/done.flag"
    s3.put_object(Bucket=bucket, Key=key, Body=b"done")
    print(f"✅ Uploaded done.flag to s3://{bucket}/{key}")

def main():
    run_mode = os.environ.get('RUN_MODE', 'TASK')
    logger.info(f"🚀 Starting field generator in {run_mode} mode")
    
    if run_mode == 'SERVICE':
        run_as_service()
    else:
        run_as_task()


def run_as_service():
    """Run as a long-lived service that checks for new work items (JSON queue first, then pair-based fallback)."""
    bucket = os.environ.get("BUCKET_NAME")
    if not bucket:
        logger.error("❗ BUCKET_NAME environment variable required for service mode")
        return

    logger.info("🔥 Starting as hot standby service (JSON-queue mode with pair-based fallback)...")

    while not shutdown_requested:
        try:
            # Primary mode: JSON work queue
            work_items = check_for_work(bucket)

            if work_items:
                for work_item in work_items:
                    if shutdown_requested:
                        logger.info("🛑 Shutdown requested, stopping work processing")
                        break
                    process_work_item(work_item, bucket)
            else:
                # Fallback: detect any unprocessed satellite/roadmap pairs
                fallback_items = detect_ready_pairs(bucket)
                if fallback_items:
                    for conn_id in fallback_items:
                        if shutdown_requested:
                            break
                        logger.info(f"🎯 Fallback processing detected pair: {conn_id}")
                        process_field_generation(conn_id, bucket)
                else:
                    logger.info("💤 No pending work found (queue or pairs), sleeping...")
                    time.sleep(30)

        except Exception as e:
            logger.error(f"❗ Error in service loop: {e}")
            time.sleep(60)

    logger.info("✅ Service shutdown complete")


def detect_ready_pairs(bucket_name):
    """Detect orphaned satellite/roadmap pairs not yet processed."""
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        all_keys = [obj["Key"] for obj in response.get("Contents", [])]

        ready_conn_ids = []
        for key in all_keys:
            if key.endswith("_satellite.png"):
                base = key[:-len("_satellite.png")]
                roadmap_key = f"{base}_roadmap.png"
                done_flag = f"outputs/{base}/done.flag"

                if roadmap_key in all_keys and done_flag not in all_keys:
                    ready_conn_ids.append(base)

        if ready_conn_ids:
            logger.info(f"📋 Detected {len(ready_conn_ids)} fallback tasks: {ready_conn_ids}")
        return ready_conn_ids

    except Exception as e:
        logger.error(f"❗ Error detecting ready pairs: {e}")
        return []

def run_as_task():
    """Original one-shot task behavior"""
    if len(sys.argv) < 2:
        print("❌ base_name argument is required")
        sys.exit(1)

    base_name = sys.argv[1]
    logger.info(f"🎯 Processing single task: {base_name}")
    process_field_generation(base_name)

def check_for_work(bucket_name):
    """Check S3 for pending field generation work"""
    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix='work-queue/field_generation/',
            MaxKeys=5  # Process up to 5 items at once
        )
        
        work_items = []
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.json'):
                try:
                    # Get the work item
                    work_response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    work_data = json.loads(work_response['Body'].read())
                    work_data['s3_key'] = obj['Key']
                    work_items.append(work_data)
                except Exception as e:
                    logger.error(f"❗ Error reading work item {obj['Key']}: {e}")
        
        if work_items:
            logger.info(f"📋 Found {len(work_items)} work items")
        
        return work_items
        
    except Exception as e:
        logger.error(f"❗ Error checking for work: {e}")
        return []

def process_work_item(work_item, bucket_name):
    """Process a single work item"""
    try:
        base_name = work_item.get('base_name')
        logger.info(f"🎯 Processing work: {base_name}")
        
        # Do the actual field generation work
        process_field_generation(base_name, bucket_name)
        
        # Mark as complete by deleting the work item
        s3.delete_object(Bucket=bucket_name, Key=work_item['s3_key'])
        logger.info(f"✅ Completed work: {base_name}")
        
    except Exception as e:
        logger.error(f"❗ Error processing work item: {e}")
        # Move to error location instead of deleting
        try:
            error_key = work_item['s3_key'].replace('work-queue/', 'work-errors/')
            s3.copy_object(
                Bucket=bucket_name,
                CopySource={'Bucket': bucket_name, 'Key': work_item['s3_key']},
                Key=error_key
            )
            s3.delete_object(Bucket=bucket_name, Key=work_item['s3_key'])
            logger.info(f"📁 Moved failed work to: {error_key}")
        except Exception as cleanup_error:
            logger.error(f"❗ Error moving failed work item: {cleanup_error}")

def process_field_generation(base_name, bucket=None):
    """Your existing field generation logic - now extracted as a function"""
    if bucket is None:
        bucket = os.environ.get("BUCKET_NAME")
    
    connection_id = base_name
    
    print(f"🔍 Using bucket: {bucket}")

    satellite_key = f"{base_name}_satellite.png"
    roadmap_key = f"{base_name}_roadmap.png"

    download_with_retry(bucket, satellite_key, "satellite.png")
    download_with_retry(bucket, roadmap_key, "roadmap.png")

    print("✅ Downloaded satellite and roadmap images")

    # Initialize S3 handler for uploading intermediate files
    s3_handler = S3Handler(connection_id, bucket)

    # Step 1: Clean the roadmap (remove underground/crosswalk artifacts)
    print("🧹 Starting map cleaning process...")
    subprocess.run(["python3", "map_cleaner.py", connection_id], check=True)
    print("✅ Map cleaning completed")

    # Step 2: Extract road mask from cleaned roadmap
    print("🛣️ Starting road mask extraction...")
    subprocess.run(["python3", "road_mask_extractor.py", connection_id], check=True)
    print("✅ Road mask extraction completed")

    # Step 3: Clean and fill gaps in road mask
    print("🧹 Starting road mask cleaning and gap filling...")
    subprocess.run(["python3", "road_mask_cleaner.py", connection_id], check=True)
    print("✅ Road mask cleaning completed")
    
    # Upload road mask cleaner outputs to S3
    print("🌐 Uploading road mask cleaner outputs to S3...")
    s3_handler.upload_road_mask_cleaner_outputs(connection_id)

    # Step 4: Generate comprehensive road network with lanes (replaces 01_generate_skelton_waypoints_lanes.py)
    print("🚗 Starting comprehensive road network generation...")
    subprocess.run(["python3", "road_network_generator.py", connection_id], check=True)
    print("✅ Road network generation completed")
    
    # Step 5: Upload all outputs to S3
    print("🌐 Starting S3 upload process...")
    uploaded_count, failed_count = s3_handler.upload_all_road_network_outputs(connection_id)
    
    if failed_count == 0:
        print("✅ All outputs successfully uploaded to S3")
    else:
        print(f"⚠️  {uploaded_count} files uploaded, {failed_count} failed")
    
    upload_done_flag(connection_id, bucket)

if __name__ == "__main__":
    main()