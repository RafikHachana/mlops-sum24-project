from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

def train(train_data, val_data):
    # Extract features and labels
    X_train = train_data.drop(columns=['Average playtime two weeks'])
    y_train = train_data['Average playtime two weeks']
    X_val = val_data.drop(columns=['Average playtime two weeks'])
    y_val = val_data['Average playtime two weeks']

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Validate the model
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

    return model

def evaluate(model, test_data):
    # Extract features and labels
    X_test = test_data.drop(columns=['Average playtime two weeks'])
    y_test = test_data['Average playtime two weeks']

    # Evaluate the model
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {test_accuracy}")

    return {"accuracy": test_accuracy}

def log_metadata(model, metrics, cfg):
    # Log model and metrics using MLflow
    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(metrics)
