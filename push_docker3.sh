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
docker build -t $REPO_NAME_1:$IMAGE_TAG $DOCKER_CONTEXT_DIR_1 --no-cache || {
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
docker build -t $REPO_NAME_2:$IMAGE_TAG $DOCKER_CONTEXT_DIR_2 --no-cache || {
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
echo ""#!/bin/bash

# --- CONFIGURATION ---
REGION="ap-northeast-1"
ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text)
IMAGE_TAG=\$(date +%Y%m%d-%H%M%S)
UNIQUE_BUILD_ID=\$(uuidgen | cut -c1-8)  # ✅ Add unique build ID

echo "🚀 AGGRESSIVE CACHE BUSTING - Build ID: \$UNIQUE_BUILD_ID"
echo "Building and deploying Docker images with tag: \$IMAGE_TAG"
echo "=================================================="

# --- DOCKER BUILD FUNCTION with AGGRESSIVE CACHE CLEARING ---
build_and_deploy() {
    local REPO_NAME=\$1
    local CLUSTER_NAME=\$2
    local SERVICE_NAME=\$3
    local DOCKER_CONTEXT_DIR=\$4
    local LABEL=\$5
    
    local ECR_URI="\$ACCOUNT_ID.dkr.ecr.\$REGION.amazonaws.com/\$REPO_NAME"
    
    echo ""
    echo "🧹 [\$LABEL] AGGRESSIVE CLEANUP - Removing ALL related images..."
    
    # ✅ Remove ALL local images related to this repo
    docker images --format "table {{.Repository}}:{{.Tag}}" | grep -E "^\${REPO_NAME}:" | xargs -r docker rmi -f 2>/dev/null || true
    docker images --format "table {{.Repository}}:{{.Tag}}" | grep -E "^\${ECR_URI}:" | xargs -r docker rmi -f 2>/dev/null || true
    
    # ✅ Prune build cache
    docker builder prune -f --all 2>/dev/null || true
    
    # ✅ System-wide cleanup
    docker system prune -f 2>/dev/null || true
    
    echo "📦 [\$LABEL] Building with ZERO cache..."
    
    # ✅ Add build args to force rebuild
    docker build \
        -t \$REPO_NAME:\$IMAGE_TAG \
        --no-cache \
        --pull \
        --build-arg CACHEBUST=\$UNIQUE_BUILD_ID \
        --build-arg BUILD_DATE=\$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        \$DOCKER_CONTEXT_DIR || {
        echo "❌ Failed to build \$LABEL"
        exit 1
    }

    echo "🔐 [\$LABEL] ECR Login..."
    aws ecr get-login-password --region \$REGION | \
      docker login --username AWS --password-stdin "\$ACCOUNT_ID.dkr.ecr.\$REGION.amazonaws.com"

    echo "🔍 [\$LABEL] ECR Repository check..."
    if ! aws ecr describe-repositories --repository-names "\$REPO_NAME" --region "\$REGION" >/dev/null 2>&1; then
      echo "📦 [\$LABEL] Creating repository \$REPO_NAME..."
      aws ecr create-repository --repository-name "\$REPO_NAME" --region "\$REGION"
    fi

    echo "🏷️ [\$LABEL] Tagging with UNIQUE tags..."
    docker tag \$REPO_NAME:\$IMAGE_TAG \$ECR_URI:\$IMAGE_TAG
    docker tag \$REPO_NAME:\$IMAGE_TAG \$ECR_URI:build-\$UNIQUE_BUILD_ID
    docker tag \$REPO_NAME:\$IMAGE_TAG \$ECR_URI:latest

    echo "📤 [\$LABEL] Pushing ALL tags..."
    docker push \$ECR_URI:\$IMAGE_TAG
    docker push \$ECR_URI:build-\$UNIQUE_BUILD_ID
    docker push \$ECR_URI:latest

    echo "🔄 [\$LABEL] FORCE ECS service update..."
    aws ecs update-service \
      --cluster \$CLUSTER_NAME \
      --service \$SERVICE_NAME \
      --force-new-deployment \
      --region \$REGION \
      --no-cli-pager

    echo "✅ [\$LABEL] Deployment complete!"
}

# --- DEPLOY FIELD GENERATOR ---
build_and_deploy \
    "field-generator" \
    "field-gen-cluster" \
    "field-gen-service" \
    "docker_fargate" \
    "FIELD-GEN"

# --- DEPLOY SVG GENERATOR ---
build_and_deploy \
    "svg-animation-generator" \
    "svg-gen-cluster" \
    "svg-gen-service" \
    "svgDocker" \
    "SVG-GEN"

# --- LAMBDA UPDATE (if needed) ---
echo ""
echo "🔧 UPDATING LAMBDA FUNCTION..."
if [ -f "analytics.py" ]; then
    echo "📦 Creating Lambda deployment package..."
    
    # Create temp directory
    TEMP_DIR=\$(mktemp -d)
    cp analytics.py \$TEMP_DIR/
    
    # Install dependencies if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -t \$TEMP_DIR/
    fi
    
    # Create zip
    cd \$TEMP_DIR
    zip -r lambda-deploy.zip .
    cd - > /dev/null
    
    # Update Lambda function
    LAMBDA_FUNCTION_NAME="your-analytics-function-name"  # ⚠️ UPDATE THIS
    aws lambda update-function-code \
        --function-name \$LAMBDA_FUNCTION_NAME \
        --zip-file fileb://\$TEMP_DIR/lambda-deploy.zip \
        --region \$REGION \
        --no-cli-pager
    
    # Cleanup
    rm -rf \$TEMP_DIR
    
    echo "✅ Lambda function updated!"
else
    echo "⚠️ No analytics.py found, skipping Lambda update"
fi

# --- VERIFICATION ---
echo ""
echo "🎉 ALL DEPLOYMENTS COMPLETE!"
echo "=================================================="
echo "🔍 VERIFICATION COMMANDS:"
echo ""
echo "# Check ECS deployments:"
echo "aws ecs describe-services --cluster field-gen-cluster --services field-gen-service --region \$REGION --query 'services[0].deployments[0]'"
echo "aws ecs describe-services --cluster svg-gen-cluster --services svg-gen-service --region \$REGION --query 'services[0].deployments[0]'"
echo ""
echo "# Check Lambda version:"
echo "aws lambda get-function --function-name your-analytics-function-name --region \$REGION --query 'Configuration.LastModified'"
echo ""
echo "# View logs:"
echo "aws logs tail /ecs/field-gen-service --follow --region \$REGION"
echo "aws logs tail /ecs/svg-gen-service --follow --region \$REGION"
echo "aws logs tail /aws/lambda/your-analytics-function-name --follow --region \$REGION"
echo "# SVG Generator logs:"
echo "aws logs tail /ecs/$SERVICE_NAME_2 --follow --region $REGION"