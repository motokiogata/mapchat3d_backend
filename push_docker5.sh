#!/usr/bin/env bash
#
# push_docker_fixed.sh
# Robust ECR build & deploy with real cache-busting and correct flag order.
# - Fixes: flags must come BEFORE the build context; backslash line breaks; stale context issues.
# - Adds: --pull, --no-cache, build arg cachebust, unique tags, ECR login, ECS redeploy.
# - Notes: Your Lambda error shows a ZIP-based function (managed runtime). Docker image updates won't affect that.
#          Update the Lambda code separately if it's not an image-based Lambda.
#
set -euo pipefail

##############
# CONFIG
##############
REGION="${REGION:-ap-northeast-1}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"   # unique tag for every deploy
UNIQUE_BUILD_ID="${UNIQUE_BUILD_ID:-$(uuidgen | cut -c1-8)}"
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"

# --- DOCKER 1: Field Generator (Fargate) ---
REPO_NAME_1="${REPO_NAME_1:-field-generator}"
ECR_URI_1="${ECR_URI_1:-$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_1}"
DOCKER_CONTEXT_DIR_1="${DOCKER_CONTEXT_DIR_1:-docker_fargate}"   # <-- ensure this includes the app code
CLUSTER_NAME_1="${CLUSTER_NAME_1:-field-gen-cluster}"
SERVICE_NAME_1="${SERVICE_NAME_1:-field-gen-service}"

# --- DOCKER 2: SVG Animation Generator (Fargate) ---
REPO_NAME_2="${REPO_NAME_2:-svg-animation-generator}"
ECR_URI_2="${ECR_URI_2:-$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME_2}"
DOCKER_CONTEXT_DIR_2="${DOCKER_CONTEXT_DIR_2:-svgDocker}"        # <-- ensure this path is correct
CLUSTER_NAME_2="${CLUSTER_NAME_2:-svg-gen-cluster}"
SERVICE_NAME_2="${SERVICE_NAME_2:-svg-gen-service}"

# Optional: Lambda ZIP function (not image-based!). If you still use ZIP packaging for 'analytics.py', set these:
#LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-}"   # e.g., analytics-backend (leave empty if not applicable)
#LAMBDA_SOURCE_DIR="${LAMBDA_SOURCE_DIR:-}"         # e.g., lambda_src (must contain analytics.py)
#LAMBDA_ZIP_PATH="${LAMBDA_ZIP_PATH:-/tmp/lambda_pkg.zip}"

##############
# PRECHECKS
##############
command -v docker >/dev/null 2>&1 || { echo "Docker not found"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "AWS CLI not found"; exit 1; }

# Ensure context dirs exist
[[ -d "$DOCKER_CONTEXT_DIR_1" ]] || { echo "Missing context: $DOCKER_CONTEXT_DIR_1"; exit 1; }
[[ -d "$DOCKER_CONTEXT_DIR_2" ]] || { echo "Missing context: $DOCKER_CONTEXT_DIR_2"; exit 1; }

##############
# ECR LOGIN
##############
echo "🔐 Logging into ECR ($REGION) ..."
aws ecr get-login-password --region "$REGION" | docker login       --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Create repos if missing
for REPO in "$REPO_NAME_1" "$REPO_NAME_2"; do
  aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 ||         aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null
done

##############
# BUILD + PUSH function
##############
build_and_push () {
  local CONTEXT="$1"
  local ECR_URI="$2"
  local IMAGE_TAG="$3"
  local UNIQUE_BUILD_ID="$4"
  local GIT_SHA="$5"

  # Important: flags BEFORE context; use --pull and --no-cache
  echo "🛠️  Building image for context: $CONTEXT -> $ECR_URI:$IMAGE_TAG"
  docker build         --pull         --no-cache         --build-arg CACHEBUST="$UNIQUE_BUILD_ID"         --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"         --build-arg GIT_SHA="$GIT_SHA"         -t "$ECR_URI:$IMAGE_TAG"         -t "$ECR_URI:latest"         "$CONTEXT"

  echo "🚀 Pushing tags: $ECR_URI:$IMAGE_TAG and :latest"
  docker push "$ECR_URI:$IMAGE_TAG"
  docker push "$ECR_URI:latest"
}

##############
# BUILD & PUSH
##############
build_and_push "$DOCKER_CONTEXT_DIR_1" "$ECR_URI_1" "$IMAGE_TAG" "$UNIQUE_BUILD_ID" "$GIT_SHA"
build_and_push "$DOCKER_CONTEXT_DIR_2" "$ECR_URI_2" "$IMAGE_TAG" "$UNIQUE_BUILD_ID" "$GIT_SHA"

##############
# ECS REDEPLOY (force new task set uses the new image digest)
##############
echo "♻️  Forcing ECS new deployments..."
aws ecs update-service --cluster "$CLUSTER_NAME_1" --service "$SERVICE_NAME_1" --force-new-deployment --region "$REGION" >/dev/null
aws ecs update-service --cluster "$CLUSTER_NAME_2" --service "$SERVICE_NAME_2" --force-new-deployment --region "$REGION" >/dev/null
echo "✅ ECS services updated (new tasks will pull latest image digest)."

##############
# OPTIONAL: LAMBDA (ZIP-BASED) UPDATE
##############
if [[ -n "${LAMBDA_FUNCTION_NAME}" && -n "${LAMBDA_SOURCE_DIR}" ]]; then
  echo "📦 Packaging Lambda ZIP for ${LAMBDA_FUNCTION_NAME} from ${LAMBDA_SOURCE_DIR} ..."
  rm -f "$LAMBDA_ZIP_PATH"
  (
    cd "$LAMBDA_SOURCE_DIR"
    # Tip: make sure analytics.py line 452 has valid syntax (elif requires a preceding if/elif)
    zip -r9 "$LAMBDA_ZIP_PATH" . >/dev/null
  )
  echo "⬆️  Updating Lambda function code (ZIP) ..."
  aws lambda update-function-code         --function-name "$LAMBDA_FUNCTION_NAME"         --zip-file "fileb://$LAMBDA_ZIP_PATH"         --region "$REGION" >/dev/null
  echo "✅ Lambda ZIP updated."
else
  echo "ℹ️  Skipping Lambda ZIP update (LAMBDA_FUNCTION_NAME or LAMBDA_SOURCE_DIR not set)."
fi

echo "🎉 Done."
echo "Tags used: $IMAGE_TAG (and latest); cachebust=$UNIQUE_BUILD_ID; git=$GIT_SHA"
