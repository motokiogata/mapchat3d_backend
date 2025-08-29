#s3_handler.py

import boto3
import json
import os
from datetime import datetime

class S3Handler:
    def __init__(self, connection_id, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.connection_id = connection_id

    def upload_to_s3(self, local_file_path, s3_key):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            print(f"✅ Uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"❌ Failed to upload {local_file_path} to S3: {e}")
            return False
    
    def upload_json_to_s3(self, data, s3_key):
        """Upload JSON data directly to S3"""
        try:
            json_string = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_string.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Uploaded JSON data to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"❌ Failed to upload JSON to S3: {e}")
            return False

    def upload_all_road_network_outputs(self, connection_id):
        """Upload all road network outputs to S3"""
        print(f"\n🌐 UPLOADING ROAD NETWORK OUTPUTS TO S3...")
        print("-" * 40)
        
        s3_prefix = f"outputs/{connection_id}/"
        uploaded_files = []
        failed_uploads = []
        
        # Define all output files to upload
        output_files = [
            f"{connection_id}_integrated_road_network.json",
            f"{connection_id}_centerlines_with_metadata.json", 
            f"{connection_id}_intersections_with_metadata.json",
            f"{connection_id}_lane_tree_routes_enhanced.json",
            f"{connection_id}_metadata_only_network.json",
            f"{connection_id}_roads_metadata_only.json",
            f"{connection_id}_intersections_metadata_only.json", 
            f"{connection_id}_lanes_metadata_only.json",
            f"{connection_id}_integrated_network_visualization.png",
            "comprehensive_narrative_network_visualization.png",
            "comprehensive_narrative_lane_visualization.png",
            f"{connection_id}_debug_skeleton.png"
        ]
        
        # Upload each file if it exists
        for filename in output_files:
            if os.path.exists(filename):
                s3_key = f"{s3_prefix}{filename}"
                
                # Determine content type
                if filename.endswith('.json'):
                    content_type = 'application/json'
                elif filename.endswith('.png'):
                    content_type = 'image/png'
                else:
                    content_type = 'binary/octet-stream'
                
                try:
                    self.s3_client.upload_file(
                        filename,
                        self.bucket_name,
                        s3_key,
                        ExtraArgs={'ContentType': content_type}
                    )
                    uploaded_files.append(filename)
                    print(f"  ✅ {filename}")
                except Exception as e:
                    failed_uploads.append((filename, str(e)))
                    print(f"  ❌ {filename}: {e}")
            else:
                print(f"  ⚠️  {filename}: File not found")
        
        print(f"\n📊 UPLOAD SUMMARY:")
        print(f"  ✅ Successfully uploaded: {len(uploaded_files)} files")
        print(f"  ❌ Failed uploads: {len(failed_uploads)} files")
        print(f"  🌐 S3 location: s3://{self.bucket_name}/{s3_prefix}")
        
        if failed_uploads:
            print(f"\n❌ FAILED UPLOADS:")
            for filename, error in failed_uploads:
                print(f"  - {filename}: {error}")
        
        # Create upload summary file
        summary = {
            "upload_timestamp": datetime.now().isoformat(),
            "connection_id": connection_id,
            "bucket": self.bucket_name,
            "s3_prefix": s3_prefix,
            "uploaded_files": uploaded_files,
            "failed_uploads": [{"file": f, "error": e} for f, e in failed_uploads],
            "total_uploaded": len(uploaded_files),
            "total_failed": len(failed_uploads)
        }
        
        # Upload summary
        summary_key = f"{s3_prefix}upload_summary.json"
        self.upload_json_to_s3(summary, summary_key)
        
        return len(uploaded_files), len(failed_uploads)