import json
import boto3
import numpy as np
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Cache embeddings in memory while Lambda is warm
cached_embeddings = None
BUCKET_NAME = os.environ.get('EMBEDDINGS_BUCKET')
EMBEDDINGS_KEY = os.environ.get('EMBEDDINGS_KEY', 'embeddings.json')

def load_embeddings():
    global cached_embeddings
    if cached_embeddings is None:
        logger.info("Loading embeddings from S3...")
        response = s3.get_object(Bucket=BUCKET_NAME, Key=EMBEDDINGS_KEY)
        cached_embeddings = json.loads(response['Body'].read())
        logger.info(f"Loaded {len(cached_embeddings)} embeddings")
    return cached_embeddings

def get_text_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": str(text)
        })
    )
    return json.loads(response["body"].read())["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def lambda_handler(event, context):
    try:
        # Get user's accident description from event
        user_text = event.get('user_text', '')
        top_k = event.get('top_k', 3)
        
        if not user_text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'user_text is required'})
            }
        
        # Load embeddings
        embeddings_data = load_embeddings()
        
        # Get embedding for user's text
        user_embedding = get_text_embedding(user_text)
        
        # Calculate similarities
        similarities = []
        for item in embeddings_data:
            similarity = cosine_similarity(user_embedding, item['embedding'])
            similarities.append({
                'case_id': item['id'],
                'similarity': similarity,
                'summary': item['summary'],
                'analysis': item['analysis']
            })
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_cases = similarities[:top_k]
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'similar_cases': top_cases,
                'total_cases_checked': len(embeddings_data)
            })
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }