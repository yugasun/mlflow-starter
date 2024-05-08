import os
from dotenv import load_dotenv


def setup_env():
    env_file = os.path.join(os.path.dirname(__file__), "../..", ".env")
    load_dotenv(dotenv_path=env_file)

    default_uri = "http://localhost:8080"
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", default_uri)

    MLFLOW_DEPLOY_HOST = os.getenv("MLFLOW_DEPLOY_HOST", "127.0.0.1")
    MLFLOW_DEPLOY_PORT = os.getenv("MLFLOW_DEPLOY_PORT", "8081")
    MLFLOW_DEPLOY_URI = f"http://{MLFLOW_DEPLOY_HOST}:{MLFLOW_DEPLOY_PORT}"

    return MLFLOW_TRACKING_URI, MLFLOW_DEPLOY_URI
