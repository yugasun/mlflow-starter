import mlflow

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# dotenv is used to load environment variables from a .env file
from config.setup_env import setup_env

# setup environment variables
MLFLOW_TRACKING_URI = setup_env()

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets. (0.75, 0.25) split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logged_model = "runs:/140c2cad5d8a4b12990536ac305b1f9d/iris_model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

# Predict on a Pandas DataFrame.
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

# output the result[:4] to the console
print(result[:4])
