#docker-entrypoint.py

import os
import sys
import boto3
from download_from_s3_with_retry import download_with_retry
import subprocess
from s3_handler import S3Handler

def upload_done_flag(connection_id, bucket):
    print(f"🔍 Attempting to upload to bucket: {bucket}")
    print(f"🔍 Key will be: outputs/{connection_id}/done.flag")
    
    s3 = boto3.client("s3")
    key = f"outputs/{connection_id}/done.flag"
    s3.put_object(Bucket=bucket, Key=key, Body=b"done")
    print(f"✅ Uploaded done.flag to s3://{bucket}/{key}")

def main():
    if len(sys.argv) < 2:
        print("❌ base_name argument is required")
        sys.exit(1)

    base_name = sys.argv[1]
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