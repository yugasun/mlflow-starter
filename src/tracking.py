import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config.setup_env import setup_env

# Load environment variables from the .env file
# Set up environment variables
MLFLOW_TRACKING_URI, _ = setup_env()
print(MLFLOW_TRACKING_URI)

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the hyperparameters of the model
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
# print(accuracy)

# Set the tracking server URI for recording, which is the MLflow tracking server
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

# Create a new MLflow experiment
mlflow.set_experiment("MLflow Quickstart1")

mlflow.autolog()

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag to remind us what this run is for
    mlflow.set_tag("Training Info", "Basic LR model for iris dataset")

    # Infer the signature of the model
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    # Load the model as a generic Python function model for prediction
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # Use the loaded model for prediction
    predictions = loaded_model.predict(X_test)

    # Get the feature names of the Iris dataset
    iris_feature_names = datasets.load_iris().feature_names

    # Create a DataFrame to store the test data and predictions
    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    # Display the first 4 rows of the result DataFrame
    result[:4]
