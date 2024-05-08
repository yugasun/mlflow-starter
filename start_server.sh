#!/bin/bash

# get env from .env file
set -a
[ -f .env ] && . .env
set +a

# Set the MLflow tracking server host and port
# get by env MLFLOW_HOST, if not set, use default: 127.0.0.1
host=${MLFLOW_HOST:-127.0.0.1}
port=${MLFLOW_PORT:-8888}

echo "Starting MLflow server at http://$host:$port"

mlflow server \
  --backend-store-uri $BACKEND_STORE_URI \
  --artifacts-destination $ARTIFACTS_DEST \
  --host $host \
  --port $port
