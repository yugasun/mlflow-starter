import mlflow
from mlflow.models import infer_signature

from sklearn import datasets
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from config.setup_env import setup_env

# setup environment variables
MLFLOW_TRACKING_URI = setup_env()

# Set tracking server uri for Logging
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

with mlflow.start_run() as run:
    # mlflow.autolog()

    # Load the diabetes dataset
    db = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train model
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset
    predictions = rf.predict(X_test)
    print(predictions)

    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)

    print(f"Run ID: {run.info.run_id}")
