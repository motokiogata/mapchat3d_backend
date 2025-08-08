import os
import boto3

ecs = boto3.client("ecs")

def lambda_handler(event, context):
    base_name = event.get("base_name")  # 例: "00032_35.67326_139.48047"
    
    response = ecs.run_task(
        cluster=os.environ['CLUSTER_NAME'],
        launchType='FARGATE',
        taskDefinition=os.environ['TASK_DEF'],
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [os.environ['SUBNET_ID']],
                'securityGroups': [os.environ['SECURITY_GROUP']],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': 'field-generator',
                    'command': [base_name]
                }
            ]
        }
    )
    
    return {
        'statusCode': 200,
        'body': f"Started Fargate task: {response['tasks'][0]['taskArn']}"
    }