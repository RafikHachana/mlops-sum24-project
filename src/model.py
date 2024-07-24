from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import mlflow
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
import pandas as pd
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skorch import NeuralNetRegressor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from mlflow.models.evaluation import evaluate

# Define Model1 and Model2 as per your code
class Model1(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.output(x)
        return x

class Model2(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.output(x)
        return x

def evaluate(model, test_data):
    X_test = test_data.drop(columns=['Average playtime two weeks'])
    y_test = test_data['Average playtime two weeks']

    # Evaluate the model
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f"Test MSE: {test_mse}")
    mlflow.log_metric("test_mse", test_mse)
    
    # Use mlflow.evaluate to log metrics and artifacts
    eval_result = mlflow.evaluate(
        model=model,
        data=X_test.to_numpy().astype(np.float32),
        targets=y_test.to_numpy().astype(np.float32),
        model_type="regressor",
        evaluators=["default"],
    )
    mlflow.log_metrics(eval_result["metrics"])

def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    import mlflow
    import pandas as pd
    import importlib
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri("http://localhost:5000")

    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_index = gs.best_index_
    best_metrics_dict = {
        key: value[best_index] for key, value in gs.cv_results_.items()
        if 'mean' in key or 'std' in key
    }

    df_train = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='Average playtime two weeks')], axis=1)
    df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='Average playtime two weeks')], axis=1)

    experiment_name = cfg['model']['model_name'] + "_" + cfg['experiment_name']
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    evaluation_metric_key = 'mean_test_score'

    run_name = "_".join([
        cfg['run_name'],
        cfg['model']['model_name'],
        cfg['model']['evaluation_metric'],
        str(best_metrics_dict[evaluation_metric_key]).replace(".", "_")
    ])

    if mlflow.active_run():
        mlflow.end_run()

    # Fake run
    # with mlflow.start_run():
    #     pass

    # Parent run
    with mlflow.start_run(run_name = run_name, experiment_id = experiment_id) as run:

        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_train, targets = cfg.data.target_cols[0]) # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_test, targets = cfg.data.target_cols[0]) # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(best_metrics_dict)
        mlflow.set_tag(cfg['model']['tag_key'], cfg['model']['tag_value'])

        X_train_sample = X_train.to_numpy()[:5].astype(np.float32)
        y_train_sample = gs.predict(X_train_sample)
        signature = infer_signature(X_train_sample, y_train_sample)

        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg['model']['artifact_path'],
            signature=signature,
            input_example=X_train_sample[0],
            registered_model_name=cfg['model']['model_name'],
            pyfunc_predict_fn=cfg['model']['pyfunc_predict_fn']
        )

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=cfg['model']['model_name'],
            version=model_info.registered_model_version,
            key="source",
            value="best_Grid_search_model"
        )

        X_test_np = X_test.to_numpy().astype(np.float32) if not isinstance(X_test, np.ndarray) else X_test.astype(np.float32)
        predictions = gs.best_estimator_.predict(X_test_np).reshape(-1)
        y_test_np = y_test.to_numpy().astype(np.float32) if not isinstance(y_test, np.ndarray) else y_test.astype(np.float32)
        y_test_np = y_test_np.reshape(-1)
        test_mse = mean_squared_error(y_test_np, predictions)
        mlflow.log_metric('test_mse', test_mse)

        eval_data = pd.DataFrame({'label': y_test_np, 'predictions': predictions})
        eval_data.to_csv('predictions.csv', index=False)
        mlflow.log_artifact('predictions.csv')
        model_count = 1
        for index, result in cv_results.iterrows():
            child_run_name = "_".join(['child', run_name, str(index)])
            with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True) as child_run:
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                module_name = cfg['model']['module_name']

                class_name_mapping = {
                    'model_1': 'Model1',
                    'model_2': 'Model2',
                }

                model_name = cfg['model']['model_name']
                class_name = class_name_mapping.get(model_name)
                if not class_name:
                    raise ValueError(f"Unknown model name: {model_name}")

                class_instance = getattr(importlib.import_module(module_name), class_name)
                model_params = {k: v for k, v in ps.items() if k in class_instance.__init__.__code__.co_varnames}
                input_dim = X_train.shape[1]
                model_params['module__input_dim'] = input_dim
                batch_size = int(ps.get('batch_size', 32))
                max_epochs = int(ps.get('max_epochs', 32))

                estimator = NeuralNetRegressor(
                    module=class_instance,
                    **model_params,
                    optimizer=optim.Adam,
                    criterion=nn.MSELoss,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    iterator_train__shuffle=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

                X_train_np = X_train.to_numpy().astype(np.float32) if not isinstance(X_train, np.ndarray) else X_train.astype(np.float32)
                y_train_np = y_train.to_numpy().astype(np.float32) if not isinstance(y_train, np.ndarray) else y_train.astype(np.float32)

                # Ensure that input data is correctly converted to tensors
                X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)

                estimator.fit(X_train_tensor, y_train_tensor)

                signature = infer_signature(X_train_np, estimator.predict(X_train_np))

                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg['model']['artifact_path'],
                    signature=signature,
                    input_example=X_train_np[0],
                    registered_model_name=cfg['model']['model_name'],
                    pyfunc_predict_fn=cfg['model']['pyfunc_predict_fn']
                )


                client.set_registered_model_alias(cfg.model.model_name, f"challenger{model_count}", model_info.registered_model_version)
                model_count += 1

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                print("Loaded model")

                X_test_np = X_test.to_numpy().astype(np.float32) if not isinstance(X_test, np.ndarray) else X_test.astype(np.float32)
                predictions = estimator.predict(X_test_np).reshape(-1)
                y_test_np = y_test_np.reshape(-1)

                # Use mlflow.evaluate to log metrics and artifacts
                # eval_result = mlflow.evaluate(
                #     model=model_uri,  # Use the model URI
                #     data=X_test_np,
                #     targets=y_test_np,
                #     model_type="regressor",
                #     evaluators=["default"],
                # )
                # mlflow.log_metrics(eval_result["metrics"])

                eval_data = pd.DataFrame({'label': y_test_np, 'predictions': predictions})
                eval_data.to_csv(f'child_predictions_{index}.csv', index=False)
                mlflow.log_artifact(f'child_predictions_{index}.csv')

                print(f"Logged child run {index} for model {class_name}")


def train(X_train, y_train, cfg):
    input_dim = X_train.shape[1]
    
    model_cfg = cfg['model']
    
    if model_cfg['model_name'] == 'model_1':
        module = Model1
    elif model_cfg['model_name'] == 'model_2':
        module = Model2
    else:
        raise ValueError("Invalid model name in configuration")
    
    net = NeuralNetRegressor(
        module=module,
        module__input_dim=input_dim,
        max_epochs=model_cfg['params']['epochs'][0],
        lr=model_cfg['params']['learning_rate'][0],
        batch_size=model_cfg['params']['batch_size'][0],
        optimizer=optim.Adam,
        criterion=nn.MSELoss,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    param_grid = {
        'lr': model_cfg['params']['learning_rate'],
        'module__dropout_rate': model_cfg['params']['dropout_rate'],
        'batch_size': model_cfg['params']['batch_size'],
        'max_epochs': model_cfg['params']['epochs']
    }

    X_train = X_train.to_numpy().astype(np.float32) if not isinstance(X_train, np.ndarray) else X_train.astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32) if not isinstance(y_train, np.ndarray) else y_train.astype(np.float32)

    scoring = make_scorer(mean_squared_error, squared=False)
    gs = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=model_cfg['cv_n_jobs'],
        refit=True,
        cv=model_cfg['folds'],
        verbose=1,
        return_train_score=True
    )


    gs.fit(X_train, y_train)


    return gs
