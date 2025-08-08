#!/bin/bash

# --- CONFIGURATION ---
REPO_NAME="field-generator"
REGION="ap-northeast-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"
IMAGE_TAG="latest"
DOCKER_CONTEXT_DIR="docker_fargate"

echo "🚀 Building Docker image..."
docker build -t $REPO_NAME:latest $DOCKER_CONTEXT_DIR || exit 1

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
docker tag $REPO_NAME:latest $ECR_URI:$IMAGE_TAG

echo "📤 Pushing image to ECR..."
docker push $ECR_URI:$IMAGE_TAG

echo "✅ Done!"
echo "📝 Use this in your ECS TaskDefinition: $ECR_URI:$IMAGE_TAG"