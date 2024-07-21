from data import extract_data # custom module
from transform_data import transform_data # custom module
from model import retrieve_model_with_alias # custom module
from utils import init_hydra # custom module
import giskard
import hydra
import mlflow


cfg = init_hydra()

version  = cfg.test_data_version

df, version = extract_data(version = version, cfg = cfg)

# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]

CATEGORICAL_COLUMNS = list(cfg.data.cat_cols) + list(cfg.data.bin_cols)

dataset_name = cfg.data.dataset_name


# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name, # Optional: Give a name to your dataset
    cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

model_name = cfg.model.best_model_name

# You can sweep over challenger aliases using Hydra
model_alias = cfg.model.best_model_alias

model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias = model_alias)  

client = mlflow.MlflowClient()

mv = client.get_model_version_by_alias(name = model_name, alias=model_alias)

model_version = mv.version

transformer_version = cfg.data_transformer_version

def predict(raw_df):
    X = transform_data(
                        df = raw_df, 
                        version = version, 
                        cfg = cfg, 
                        return_df = False, 
                        only_transform = True, 
                        transformer_version = transformer_version, 
                        only_X = True
                      )

    return model.predict(X)

predictions = predict(df[df.columns].head())
print(predictions)

giskard_model = giskard.Model(
  model=predict,
  model_type = "classification", # regression
  classification_labels=list(cfg.data.labels),  # The order MUST be identical to the prediction_function's output order
  feature_names = df.columns, # By default all columns of the passed dataframe
  name=model_name, # Optional: give it a name to identify it in metadata
  # classification_threshold=0.5, # Optional: Default: 0.5
)

scan_results = giskard.scan(giskard_model, giskard_dataset)

# Save the results in `html` file
scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
scan_results.to_html(scan_results_path)

suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
test_suite = giskard.Suite(name = suite_name)

test1 = giskard.testing.test_f1(model = giskard_model, 
                                dataset = giskard_dataset,
                                threshold=cfg.model.f1_threshold)

test_suite.add_test(test1)

test_results = test_suite.run()
if (test_results.passed):
    print("Passed model validation!")
else:
    print("Model has vulnerabilities!")
