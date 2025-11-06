#!/bin/bash
# Check ECS service status

STACK_NAME="\${STACK_NAME:-\$(basename \$(pwd))}"
CLUSTER_NAME="\${STACK_NAME}-FargateCluster"

echo "🔍 Checking ECS services for stack: \$STACK_NAME"
echo ""

# Check field generator service
FIELD_SERVICE="\${STACK_NAME}-field-generator-service"
echo "📊 Field Generator Service:"
aws ecs describe-services \
  --cluster \$CLUSTER_NAME \
  --services \$FIELD_SERVICE \
  --query 'services[0].{Name:serviceName,Status:status,Running:runningCount,Desired:desiredCount,Pending:pendingCount}' \
  --output table

echo ""

# Check SVG animation service
SVG_SERVICE="\${STACK_NAME}-svg-animation-service"
echo "📊 SVG Animation Service:"
aws ecs describe-services \
  --cluster \$CLUSTER_NAME \
  --services \$SVG_SERVICE \
  --query 'services[0].{Name:serviceName,Status:status,Running:runningCount,Desired:desiredCount,Pending:pendingCount}' \
  --output table

echo ""
echo "✅ Service status check complete"