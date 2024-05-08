#!/bin/sh

# get env from ../../.env file
set -a
[ -f ../../.env ] && . ../../.env
set +a

# Set the MLflow deployment server host and port
host=${MLFLOW_DEPLOY_HOST:-127.0.0.1}
port=${MLFLOW_DEPLOY_PORT:-8081}
workers=${MLFLOW_DEPLOY_WORKERS:-1}

export OPENAI_API_KEY=${OPENAI_API_KEY}
export OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}

mlflow deployments start-server --config-path ./config.yaml --port ${port} --host ${host} --workers ${workers}
