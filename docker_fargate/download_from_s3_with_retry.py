# download_from_s3_with_retry.py
import time
import boto3
from botocore.exceptions import ClientError

def download_with_retry(bucket, key, dest_path, retries=5, delay=1.0):
    s3 = boto3.client("s3")
    for i in range(retries):
        try:
            s3.download_file(bucket, key, dest_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                time.sleep(delay)
            else:
                raise
    raise FileNotFoundError(f"S3 object {key} not found after {retries} retries")