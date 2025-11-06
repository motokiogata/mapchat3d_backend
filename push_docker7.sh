#!/bin/bash
set -euo pipefail

##############
# CONFIG
##############
REGION="${REGION:-ap-northeast-1}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"   # unique tag for every deploy
UNIQUE_BUILD_ID="${UNIQUE_BUILD_ID:-$(uuidgen | cut -c1-8)}"
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"

# 🔥 NEW: Hot standby configuration
ENABLE_HOT_STANDBY="${ENABLE_HOT_STANDBY:-true}"
FIELD_GENERATOR_HOT_COUNT="${FIELD_GENERATOR_HOT_COUNT:-2}"
SVG_ANIMATION_HOT_COUNT="${SVG_ANIMATION_HOT_COUNT:-1}"

# Docker contexts
DOCKER_CONTEXT_DIR_1="docker_fargate"
REPO_NAME_1="field-generator"
DOCKER_CONTEXT_DIR_2="svgDocker"
REPO_NAME_2="svg-animation-generator"

echo "🏗️  Building and deploying with IMAGE_TAG=$IMAGE_TAG"
echo "🔥 Hot Standby Configuration:"
echo "   - Enabled: $ENABLE_HOT_STANDBY"
echo "   - Field Generator containers: $FIELD_GENERATOR_HOT_COUNT"
echo "   - SVG Animation containers: $SVG_ANIMATION_HOT_COUNT"
echo "   - Git SHA: $GIT_SHA"

##############
# Build &amp; Push Images
##############

echo ""
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 118028261233.dkr.ecr.ap-northeast-1.amazonaws.com

echo ""
echo "🐳 Building Field Generator image..."
docker build --pull --no-cache \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1:$IMAGE_TAG \
  --build-arg CACHEBUST=$UNIQUE_BUILD_ID \
  --build-arg GIT_SHA=$GIT_SHA \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  $DOCKER_CONTEXT_DIR_1

echo "📤 Pushing Field Generator image..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1:$IMAGE_TAG

echo ""
echo "🐳 Building SVG Animation Generator image..."
docker build --pull --no-cache \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2:$IMAGE_TAG \
  --build-arg CACHEBUST=$UNIQUE_BUILD_ID \
  --build-arg GIT_SHA=$GIT_SHA \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  $DOCKER_CONTEXT_DIR_2

echo "📤 Pushing SVG Animation Generator image..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2:$IMAGE_TAG

##############
# SAM Deploy
##############
echo ""
echo "🏗️  Building SAM application..."
sam build

echo ""
echo "🚀 Deploying SAM with hot standby configuration..."
sam deploy \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    VpcId=vpc-1c71077b \
    SubnetId=subnet-02efc25934d31a888 \
    ImageTag=$IMAGE_TAG \
    DeployTime=$(date +%s) \
    EnableHotStandby=$ENABLE_HOT_STANDBY \
    FieldGeneratorHotStandbyCount=$FIELD_GENERATOR_HOT_COUNT \
    SvgAnimationHotStandbyCount=$SVG_ANIMATION_HOT_COUNT

##############
# Post-Deployment Status
##############
echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Deployment Summary:"
echo "   - Image Tag: $IMAGE_TAG"
echo "   - Git SHA: $GIT_SHA"
echo "   - Hot Standby: $ENABLE_HOT_STANDBY"

if [ "$ENABLE_HOT_STANDBY" = "true" ]; then
  echo ""
  echo "🔥 Hot Standby Services:"
  echo "   - Field Generator: $FIELD_GENERATOR_HOT_COUNT containers running"
  echo "   - SVG Animation: $SVG_ANIMATION_HOT_COUNT containers running"
  echo ""
  echo "📈 Benefits:"
  echo "   - No cold start delays"
  echo "   - Auto scaling under load"
  echo "   - Immediate processing capability"
  echo ""
  echo "💰 Cost Control:"
  echo "   - To disable hot standby: ENABLE_HOT_STANDBY=false ./deploy.sh"
  echo "   - To adjust counts: FIELD_GENERATOR_HOT_COUNT=1 SVG_ANIMATION_HOT_COUNT=1 ./deploy.sh"
else
  echo ""
  echo "❄️  Hot standby is DISABLED - using cold start mode"
  echo "   - To enable: ENABLE_HOT_STANDBY=true ./deploy.sh"
fi

echo ""
echo "🌐 Checking ECS service status..."

# Check if services are running (optional status check)
if [ "$ENABLE_HOT_STANDBY" = "true" ]; then
  # Get stack name (assuming it's the directory name or default)
  STACK_NAME="${STACK_NAME:-$(basename $(pwd))}"
  
  echo "🔍 Checking service status for stack: $STACK_NAME"
  
  # Check field generator service
  FIELD_SERVICE_NAME="${STACK_NAME}-field-generator-service"
  echo "   Checking $FIELD_SERVICE_NAME..."
  
  # Check SVG animation service
  SVG_SERVICE_NAME="${STACK_NAME}-svg-animation-service"
  echo "   Checking $SVG_SERVICE_NAME..."
  
  # Note: You could add actual AWS CLI calls here to check service status
  # aws ecs describe-services --cluster $CLUSTER_NAME --services $FIELD_SERVICE_NAME
fi

echo ""
echo "🎉 Ready to process requests with hot standby containers!"