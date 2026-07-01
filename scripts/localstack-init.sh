#!/usr/bin/env bash
set -euo pipefail

BUCKET_NAME="${S3_BUCKET_NAME:-self-rag-documents}"
QUEUE_NAME="${SQS_QUEUE_NAME:-self-rag-ingest}"
DLQ_NAME="${QUEUE_NAME}-dlq"
REGION="${AWS_REGION:-us-east-1}"

awslocal s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"

DLQ_URL=$(awslocal sqs create-queue --queue-name "$DLQ_NAME" --region "$REGION" --query QueueUrl --output text)
DLQ_ARN=$(awslocal sqs get-queue-attributes --queue-url "$DLQ_URL" --attribute-names QueueArn --region "$REGION" --query Attributes.QueueArn --output text)

awslocal sqs create-queue \
  --queue-name "$QUEUE_NAME" \
  --region "$REGION" \
  --attributes "{\"RedrivePolicy\":\"{\\\"deadLetterTargetArn\\\":\\\"${DLQ_ARN}\\\",\\\"maxReceiveCount\\\":\\\"5\\\"}\"}"

echo "localstack init complete: bucket=$BUCKET_NAME queue=$QUEUE_NAME dlq=$DLQ_NAME"
