"""
This script is used for the model validation on the Github CI
"""
import sys
import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import extract_data, transform_data # custom module
# from transform_data import transform_data # custom module
# from model import retrieve_model_with_alias # custom module
# from utils import init_hydra # custom module

import giskard
import giskard.models.cache
import hydra
import mlflow
from omegaconf import DictConfig

# BASE_PATH = "/home/rafik/Documents/InnoUni/sum24/mlops/mlops-sum24-project"

giskard.models.cache.disable_cache()
# cfg = init_hydra()
@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    version  = "v4"
    # version  = cfg.test_data_version
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df, _ = extract_data(BASE_PATH)
    df = df.sample(frac = 0.001, random_state = 88)
    df.dropna(subset=['Categories', 'Genres', 'Tags', 'Movies'], inplace=True)


    # Specify categorical columns and target column
    # TARGET_COLUMN = cfg.data.target_cols[0]
    TARGET_COLUMN = "Average playtime two weeks"

    # CATEGORICAL_COLUMNS = list(cfg.data.cat_cols) + list(cfg.data.bin_cols)

    # dataset_name = cfg.data.dataset_name
    dataset_name = "Games"


    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name, # Optional: Give a name to your dataset
        # cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
    )

    model_name = "linear_regression"

    # # You can sweep over challenger aliases using Hydra
    # model_alias = cfg.model.best_model_alias

    # model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)  

    # client = mlflow.MlflowClient()

    # mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)

    # model_version = mv.version

    # transformer_version = cfg.data_transformer_version


    # client = mlflow.MlflowClient()


    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))
    
    
    # model = challenger_models[0][1]
    def predict(raw_df):
        # print("COOOOLS", raw_df.head())
        X = transform_data(raw_df, only_x=True)#[raw_df.columns[:-1]]#.drop(columns="Average playtime two weeks")
        if 'metric' in X.columns:
            X.drop(columns='metric', inplace=True)

        # print(X.columns)

        return model.predict(X)

    predictions = predict(df.drop(columns='Average playtime two weeks').head())
    # print(predictions)

    giskard_model = giskard.Model(
    model=predict,
    model_type = "regression", # regression
    # classification_labels=list(cfg.data.labels),  # The order MUST be identical to the prediction_function's output order
    feature_names = df.drop(columns=[TARGET_COLUMN]).columns, # By default all columns of the passed dataframe
    name=model_name, # Optional: give it a name to identify it in metadata
    # classification_threshold=0.5, # Optional: Default: 0.5
    )


    # exit()
    scan_results = giskard.scan(giskard_model, giskard_dataset)

    model_version = "v0.0.1"

    # Save the results in `html` file
    scan_results_path = os.path.join(BASE_PATH, f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html")
    
    scan_results.to_html(scan_results_path)


    suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
    test_suite = giskard.Suite(name = suite_name)

    test1 = giskard.testing.test_mae(model = giskard_model, 
                                    dataset = giskard_dataset,
                                    threshold=1e40)

    test_suite.add_test(test1)

    test_results = test_suite.run()
    if (test_results.passed):
        print("Passed model validation!")
    else:
        print("Model has vulnerabilities!")
        exit(1)



if __name__ == "__main__":
    main()
