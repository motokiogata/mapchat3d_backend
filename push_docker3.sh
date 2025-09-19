#!/bin/bash

# --- CONFIGURATION ---
REGION="ap-northeast-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)

# --- DOCKER 1: Field Generator (Existing) ---
REPO_NAME_1="field-generator"
CLUSTER_NAME_1="field-gen-cluster"  # ⚠️ REPLACE with your actual cluster name
SERVICE_NAME_1="field-gen-service"  # ⚠️ REPLACE with your actual service name
ECR_URI_1="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1"
DOCKER_CONTEXT_DIR_1="docker_fargate"

# --- DOCKER 2: SVG Animation Generator (New) ---
REPO_NAME_2="svg-animation-generator"
CLUSTER_NAME_2="svg-gen-cluster"  # ⚠️ REPLACE with your SVG cluster name
SERVICE_NAME_2="svg-gen-service"  # ⚠️ REPLACE with your SVG service name
ECR_URI_2="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2"
DOCKER_CONTEXT_DIR_2="svgDocker"

echo "🚀 Building and deploying Docker images with tag: $IMAGE_TAG"
echo "=================================================="

# --- BUILD AND DEPLOY DOCKER 1: Field Generator ---
echo ""
echo "📦 [DOCKER 1] Building Field Generator..."
docker build -t $REPO_NAME_1:$IMAGE_TAG $DOCKER_CONTEXT_DIR_1 || {
    echo "❌ Failed to build Docker 1"
    exit 1
}

echo "🔐 [DOCKER 1] Logging in to Amazon ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "🔍 [DOCKER 1] Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names "$REPO_NAME_1" --region "$REGION" >/dev/null 2>&1; then
  echo "📦 [DOCKER 1] Repository not found. Creating repository $REPO_NAME_1..."
  aws ecr create-repository --repository-name "$REPO_NAME_1" --region "$REGION"
else
  echo "📦 [DOCKER 1] Repository already exists."
fi

echo "🏷️ [DOCKER 1] Tagging image..."
docker tag $REPO_NAME_1:$IMAGE_TAG $ECR_URI_1:$IMAGE_TAG
docker tag $REPO_NAME_1:$IMAGE_TAG $ECR_URI_1:latest

echo "📤 [DOCKER 1] Pushing image to ECR..."
docker push $ECR_URI_1:$IMAGE_TAG
docker push $ECR_URI_1:latest

echo "🔄 [DOCKER 1] Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME_1 \
  --service $SERVICE_NAME_1 \
  --force-new-deployment \
  --region $REGION

echo "✅ [DOCKER 1] Field Generator deployment complete!"

# --- BUILD AND DEPLOY DOCKER 2: SVG Animation Generator ---
echo ""
echo "🎬 [DOCKER 2] Building SVG Animation Generator..."
docker build -t $REPO_NAME_2:$IMAGE_TAG $DOCKER_CONTEXT_DIR_2 || {
    echo "❌ Failed to build Docker 2"
    exit 1
}

echo "🔍 [DOCKER 2] Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names "$REPO_NAME_2" --region "$REGION" >/dev/null 2>&1; then
  echo "📦 [DOCKER 2] Repository not found. Creating repository $REPO_NAME_2..."
  aws ecr create-repository --repository-name "$REPO_NAME_2" --region "$REGION"
else
  echo "📦 [DOCKER 2] Repository already exists."
fi

echo "🏷️ [DOCKER 2] Tagging image..."
docker tag $REPO_NAME_2:$IMAGE_TAG $ECR_URI_2:$IMAGE_TAG
docker tag $REPO_NAME_2:$IMAGE_TAG $ECR_URI_2:latest

echo "📤 [DOCKER 2] Pushing image to ECR..."
docker push $ECR_URI_2:$IMAGE_TAG
docker push $ECR_URI_2:latest

echo "🔄 [DOCKER 2] Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME_2 \
  --service $SERVICE_NAME_2 \
  --force-new-deployment \
  --region $REGION

echo "✅ [DOCKER 2] SVG Animation Generator deployment complete!"

# --- SUMMARY ---
echo ""
echo "🎉 ALL DEPLOYMENTS COMPLETE!"
echo "=================================================="
echo "📝 Field Generator: $ECR_URI_1:$IMAGE_TAG"
echo "📝 SVG Generator: $ECR_URI_2:$IMAGE_TAG"
echo ""
echo "🔍 Check deployment status with:"
echo "# Field Generator:"
echo "aws ecs describe-services --cluster $CLUSTER_NAME_1 --services $SERVICE_NAME_1 --region $REGION --query 'services[0].deployments'"
echo ""
echo "# SVG Animation Generator:"
echo "aws ecs describe-services --cluster $CLUSTER_NAME_2 --services $SERVICE_NAME_2 --region $REGION --query 'services[0].deployments'"
echo ""
echo "📊 View logs:"
echo "# Field Generator logs:"
echo "aws logs tail /ecs/$SERVICE_NAME_1 --follow --region $REGION"
echo ""
echo "# SVG Generator logs:"
echo "aws logs tail /ecs/$SERVICE_NAME_2 --follow --region $REGION"