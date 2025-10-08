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

# Docker contexts
DOCKER_CONTEXT_DIR_1="docker_fargate"
REPO_NAME_1="field-generator"
DOCKER_CONTEXT_DIR_2="svgDocker"
REPO_NAME_2="svg-animation-generator"

##############
# Build & Push Images
##############

# Login to ECR
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin 118028261233.dkr.ecr.ap-northeast-1.amazonaws.com

# Field Generator

docker build --pull --no-cache \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1:$IMAGE_TAG \
  --build-arg CACHEBUST=$UNIQUE_BUILD_ID \
  --build-arg GIT_SHA=$GIT_SHA \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  $DOCKER_CONTEXT_DIR_1

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1:$IMAGE_TAG

# SVG Animation Generator
docker build --pull --no-cache \
  -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2:$IMAGE_TAG \
  --build-arg CACHEBUST=$UNIQUE_BUILD_ID \
  --build-arg GIT_SHA=$GIT_SHA \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  $DOCKER_CONTEXT_DIR_2

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2:$IMAGE_TAG

##############
# SAM Deploy
##############
sam build

sam deploy \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    VpcId=vpc-1c71077b \
    SubnetId=subnet-02efc25934d31a888 \
    ImageTag=$IMAGE_TAG \
    DeployTime=$(date +%s)