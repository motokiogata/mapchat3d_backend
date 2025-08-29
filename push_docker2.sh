#!/bin/bash

# --- CONFIGURATION ---
REPO_NAME="field-generator"
REGION="ap-northeast-1"
CLUSTER_NAME="your-cluster-name"  # ⚠️ REPLACE with your actual cluster name
SERVICE_NAME="your-service-name"  # ⚠️ REPLACE with your actual service name
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
DOCKER_CONTEXT_DIR="docker_fargate"

echo "🚀 Building Docker image with tag: $IMAGE_TAG"
docker build -t $REPO_NAME:$IMAGE_TAG $DOCKER_CONTEXT_DIR || exit 1

echo "🔐 Logging in to Amazon ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "🔍 Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "📦 Repository not found. Creating repository $REPO_NAME..."
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION"
else
  echo "📦 Repository already exists."
fi

echo "🏷️ Tagging image..."
docker tag $REPO_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
docker tag $REPO_NAME:$IMAGE_TAG $ECR_URI:latest

echo "📤 Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG
docker push $ECR_URI:latest

echo "🔄 Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment \
  --region $REGION

echo "✅ Done!"
echo "📝 New image: $ECR_URI:$IMAGE_TAG"
echo "📝 Also tagged as: $ECR_URI:latest"
echo ""
echo "🔍 Check deployment status with:"
echo "aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION --query 'services[0].deployments'"