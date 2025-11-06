#!/bin/bash
# Scale ECS services manually

if [ \$# -ne 2 ]; then
    echo "Usage: \$0 <span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;field-count&gt;</span><span style="color: black; font-weight: normal;"> "
    echo "Example: \$0 3 2"
    exit 1
fi

FIELD_COUNT=\$1
SVG_COUNT=\$2
STACK_NAME="\${STACK_NAME:-\$(basename \$(pwd))}"
CLUSTER_NAME="\${STACK_NAME}-FargateCluster"

echo "🔧 Scaling services:"
echo "   Field Generator: \$FIELD_COUNT containers"
echo "   SVG Animation: \$SVG_COUNT containers"

# Scale field generator service
FIELD_SERVICE="\${STACK_NAME}-field-generator-service"
aws ecs update-service \
  --cluster \$CLUSTER_NAME \
  --service \$FIELD_SERVICE \
  --desired-count \$FIELD_COUNT

# Scale SVG animation service
SVG_SERVICE="\${STACK_NAME}-svg-animation-service"
aws ecs update-service \
  --cluster \$CLUSTER_NAME \
  --service \$SVG_SERVICE \
  --desired-count \$SVG_COUNT

echo "✅ Scaling commands sent. Services will update shortly."