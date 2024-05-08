import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# from pprint import pprint

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow import MlflowClient
from config.setup_env import setup_env

# setup environment variables
MLFLOW_TRACKING_URI = setup_env()

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def create_experiment():
    # Provide on Experiment description that will appear in the UI
    experiment_description = (
        "This is the grocery forecasting project."
        "This experiment contains the produce models for apples."
    )

    # Provide searchable tags that define characteristics of the Runs
    # that will be in this Experiment
    experiment_tags = {
        "project_name": "grocery-forecasting",
        "store_dept": "produce",
        "team": "stores-ml",
        "project_quarter": "Q3-2023",
        "mlflow.note.content": experiment_description,
    }

    # Create the Experiment, providing a unique name
    client.create_experiment(name="Apple_Models", tags=experiment_tags)

    # Use search_experiments() to search on the project_name tag key
    apples_experiment = client.search_experiments(
        filter_string="tags.`project_name` = 'grocery-forecasting'"
    )
    print(vars(apples_experiment[0]))


def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000, n_rows: int = 5000
):
    """
    Generates a synthetic dataset for predicting apple sales demand with seasonality
      and inflation.

      This function creates a pandas DataFrame with features relevant to apple sales.
      The features include date, average_temperature, rainfall, weekend flag, holiday flag,
      promotional flag, price_per_kg, and the previous day's demand. The target variable,
      'demand', is generated based on a combination of these features with some added noise.

      Args:
          base_demand (int, optional): Base demand for apples. Defaults to 1000.
          n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.

      Returns:
          pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

      Example:
          >>> df = generate_apple_sales_data_with_seasonality(base_demand=1200, n_rows=6000)
          >>> df.head()
    """

    # Set seed for reproducibility
    np.random.seed(9999)

    # Create date range
    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()

    # Generate features
    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows),
            "rainfall": np.random.exponential(5, n_rows),
            "weekend": [(date.weekday() >= 5) for date in dates],
            "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows),
            "month": [date.month for date in dates],
        }
    )

    # Introduce inflation over time (years)
    df["inflation_multiplier"] = (
        1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
    )

    # Incorporate seasonality due to apple harvests
    df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
        2 * np.pi * (df["month"] - 9) / 12
    )

    # Modify the price_per_kg based on the harvest effect
    df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

    # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
    peak_month = [4, 10]
    df["promo"] = np.where(
        df["month"].isin(peak_month),
        1,
        np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    )

    # Generate target variable based on features
    base_price_effect = -df["price_per_kg"] * 50
    seasonality_effect = df["harvest_effect"] * 50
    promo_effect = df["promo"] * 200

    df["demand"] = (
        base_demand
        + base_price_effect
        + seasonality_effect
        + promo_effect
        + df["weekend"] * 300
        + np.random.normal(0, 50, n_rows)
    ) * df["inflation_multiplier"]  # add random noise

    # Add previous day's demand
    df["previous_days_demand"] = df["demand"].shift(1)
    df["previous_days_demand"].fillna(
        method="bfill", inplace=True
    )  # fill the first row

    # Drop temporary columns
    df.drop(
        columns=["month", "harvest_effect", "inflation_multiplier"],
        inplace=True,
    )

    return df


create_experiment()

# Sets the current active experiment to the "Apple_Models" experiment and
# returns the Experiment metadata
apple_experiment = mlflow.set_experiment("Apple_Models")

# Define a run name for this iteration of training
# If this is not set, a unique name will be auto-generated for your run.
run_name = "apples_rf_test"

# Define on artifact path that the model will be saved to
artifact_path = "rf_apples"

data = generate_apple_sales_data_with_promo_adjustment()
# pprint(data)

# Initiate the MLflow run context
with mlflow.start_run(run_name=run_name) as run:
    # Split the data into features and target and drop irrelevant date field and target field
    X = data.drop(columns=["date", "demand"])
    y = data["demand"]

    # Split the date into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 888,
    }

    # Train the RandomForestRegressor
    rf = RandomForestRegressor(**params)

    # Fit the model on the training data
    rf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = rf.predict(X_val)

    # Calculate error metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    # Assemble the metrics we're going to write into a collection
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    # Log on instance of the trained model for later use
    mlflow.sklearn.log_model(
        sk_model=rf, input_example=X_val, artifact_path=artifact_path
    )
