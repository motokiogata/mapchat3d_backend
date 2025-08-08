# docker-entrypoint.py
import os
import sys
import boto3
from download_from_s3_with_retry import download_with_retry
import subprocess

def upload_done_flag(connection_id, bucket):  # ← Remove default value
    print(f"🔍 Attempting to upload to bucket: {bucket}")  # ← Add this debug line
    print(f"🔍 Key will be: outputs/{connection_id}/done.flag")  # ← Add this debug line
    
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
    
    print(f"🔍 Using bucket: {bucket}")  # ← Add this debug line

    satellite_key = f"{base_name}_satellite.png"
    roadmap_key = f"{base_name}_roadmap.png"

    download_with_retry(bucket, satellite_key, "satellite.png")
    download_with_retry(bucket, roadmap_key, "roadmap.png")

    print("✅ Downloaded satellite and roadmap images")


    subprocess.run(["python3", "00_01_extract_road_mask_from_map9.py", connection_id], check=True)
    subprocess.run(["python3", "00_02_clean_road_mask.py", connection_id], check=True)
    subprocess.run(["python3", "01_generate_skelton_waypoints_lanes.py", connection_id], check=True)
    
    upload_done_flag(connection_id, bucket)  # ← Pass the bucket parameter!

if __name__ == "__main__":
    main()