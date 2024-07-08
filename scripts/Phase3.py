import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions
import numpy as np
import great_expectations as gx
import multiprocessing as mp

# Load your dataset
dataset = pd.read_csv('games.csv')

# Initialize encoders
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
label_encoder = LabelEncoder()

# Define a helper function to infer and parse date columns with a specific format
def infer_and_parse_date(column, format=None):
    try:
        return pd.to_datetime(column, format=format, errors='coerce')
    except ValueError:
        return pd.to_datetime(column, errors='coerce')

# Process each column based on the criteria
for column in dataset.columns:
    if pd.api.types.is_numeric_dtype(dataset[column]):
        continue  # Skip numeric columns
    
    unique_values = dataset[column].nunique()
    
    if unique_values <= 5:
        encoded_data = one_hot_encoder.fit_transform(dataset[[column]])
        encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{cat}" for cat in one_hot_encoder.categories_[0][1:]])
        dataset = pd.concat([dataset.drop(columns=[column]), encoded_df], axis=1)
    elif unique_values <= 10:
        dataset[column] = label_encoder.fit_transform(dataset[column])
    else:
        dataset = dataset.drop(columns=[column])

# Drop non-numerical columns that are still left
dataset_numerical = dataset.select_dtypes(include=[int, float])

# Fill missing data with the mean of each column
dataset_numerical = dataset_numerical.fillna(dataset_numerical.mean())

# Define the target variable and features
target_column = "Average playtime two weeks"
X = dataset_numerical.drop(columns=[target_column])
y = dataset_numerical[target_column]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up Great Expectations context
context = gx.get_context(context_root_dir="gx")

# Define the Pandas data source
ds1 = context.sources.add_or_update_pandas(name="my_pandas_ds")

# Create a Data Asset from the dataset
da1 = ds1.add_dataframe_asset(name="playtime_asset")

# Build a batch request
batch_request = da1.build_batch_request(dataframe=X)

# Create an Expectation Suite
expectation_suite_name = "playtime_expectations"
context.add_or_update_expectation_suite(expectation_suite_name)

# Create a Validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=expectation_suite_name
)

# Add Expectations based on the provided information
# Example for "Metacritic score"
validator.expect_column_values_to_be_between(
    column="Metacritic score",
    min_value=0,
    max_value=100
)

# Example for "User score"
validator.expect_column_values_to_be_between(
    column="User score",
    min_value=0,
    max_value=100
)

# Add more expectations based on the provided data characteristics...

# Save the expectation suite
validator.save_expectation_suite(discard_failed_expectations=False)

# Create a checkpoint
checkpoint_name = "playtime_checkpoint"
checkpoint = context.add_or_update_checkpoint(
    name=checkpoint_name,
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        },
    ],
)

# Run the checkpoint
checkpoint_result = checkpoint.run()
assert checkpoint_result.success, "Checkpoint validation failed!"

# Function to run MLflow experiment
def run_experiment(args):
    params, model_class, model_name, experiment_id = args
    model = model_class()
    
    mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run(run_name=f"{model_name} run") as run:
        # Train the model
        model.set_params(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the performance metrics
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)

        # Set a tag for the run
        mlflow.set_tag("Model", model_name)

        # Infer the model signature
        signature = infer_signature(X_test, y_pred)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            signature=signature,
            input_example=X_test,
            registered_model_name=f"{model_name}_Playtime_Regressor"
        )

        print(f'{model_name} - Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}')

# Set MLflow tracking URI to the local server
mlflow.set_tracking_uri("http://localhost:5000")

experiment_name = "Playtime Prediction Experiment"
try:
    # Create a new MLflow Experiment
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Define hyperparameter grids
random_forest_params_list = [
    {"n_estimators": 100, "random_state": 42},
    {"n_estimators": 200, "random_state": 42},
    {"n_estimators": 100, "max_depth": 10, "random_state": 42}
]

gradient_boosting_params_list = [
    {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
    {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42},
    {"n_estimators": 100, "learning_rate": 0.01, "random_state": 42}
]

# Combine the parameters with the corresponding model classes and names
experiments = [(params, RandomForestRegressor, "RandomForestRegressor", experiment_id) for params in random_forest_params_list]
experiments += [(params, GradientBoostingRegressor, "GradientBoostingRegressor", experiment_id) for params in gradient_boosting_params_list]

# Run experiments using multiprocessing

with mp.Pool(processes=4) as pool:
    pool.map(run_experiment, experiments)
