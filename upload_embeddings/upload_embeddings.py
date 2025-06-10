import json
import boto3
import base64
import gzip
import cfnresponse

s3 = boto3.client('s3')

# Your embeddings data - you can either embed it here or read from a file
# For large files, it's better to read from file during build
def get_embeddings_data():
    # Option 1: Read from file (put embeddings.json in upload_embeddings/ folder)
    try:
        with open('embeddings.json', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Option 2: Return empty data or error
        return None

def lambda_handler(event, context):
    try:
        request_type = event['RequestType']
        
        if request_type in ['Create', 'Update']:
            bucket_name = event['ResourceProperties']['BucketName']
            
            # Get embeddings data
            embeddings_data = get_embeddings_data()
            if not embeddings_data:
                cfnresponse.send(event, context, cfnresponse.FAILED, {}, reason="No embeddings data found")
                return
            
            # Upload to S3
            s3.put_object(
                Bucket=bucket_name,
                Key='embeddings.json',
                Body=embeddings_data,
                ContentType='application/json'
            )
            
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                'Message': 'Embeddings uploaded successfully'
            })
        else:
            # Delete - optionally remove the file
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            
    except Exception as e:
        print(f"Error: {e}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {}, reason=str(e))