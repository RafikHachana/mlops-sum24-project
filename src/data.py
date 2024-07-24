
import pandas as pd
import json
import gc
import great_expectations as gx
from great_expectations.validator.validator import Validator
from great_expectations.data_context import FileDataContext
import re
import yaml
from great_expectations.dataset import PandasDataset

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import hydra
from omegaconf import DictConfig
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from zenml.client import Client
from sklearn.model_selection import train_test_split
import zenml


def load_features_training(name, version, size = 1):
    client = Client()
    l = client.list_artifact_versions(name = name, tag = version, sort_by="version").items
    # print(l)
    l.sort(key=lambda x: int(x.version), reverse=True)

    # print("Choosing this artifact", [x.id for x in l])

    # print("list of artifacts: ", l)

    # df = client.get_artifact_version('99ce7f88-b396-42c0-8e0d-86a576011216').load()
    df = l[0].load()
    if size < 1:
        df = df.sample(frac = size, random_state = 88)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y


# def extract_data_training(cfg):
#     # Fetch the ZenML artifact store client
#     client = Client()


#     data = client.list_artifact_versions(name ="features_target", sort_by="version").items
#     data.reverse()
#     data = data[0].load()

#     print("NaN", data.isna().sum().sum())
#     # y = df['Average playtime two weeks']
#     # X = df.drop(columns=['Average playtime two weeks'])
#     # y.reverse()
#     # y = y[0].load()

#     # Load the data sample based on the version
#     # data_version = cfg.data_version
#     # artifact = artifact_store.get_artifact(name=f"data_sample_{data_version}")
#     # data = pd.read_csv(artifact.uri)

#     # Split data into training and validation sets
#     train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

#     # Load the test data sample based on the version
#     # test_data_version = cfg.test_data_version
#     # test_artifact = artifact_store.get_artifact(name=f"data_sample_{test_data_version}")
#     # test_data = pd.read_csv(test_artifact.uri)

#     # TODO: What is this?
#     # Split the test data
#     _, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

#     return train_data, val_data, test_data


URL_REGEX = r"^(?:https?:\/\/)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$"

def transform_data(df, only_x=False):
    # Your data transformation code

    gc.collect()
    # DROP
    # drop Header image
    # drop Score rank. It has 71674 null values (out of 71716!).
    # probably drop Develops (too many unique values 42615). We can also transform these features to has_experienced_developer (more than 50 games)
    # probably drop Publishers (too many unique values 36815). We can also transform these features to has_experienced_publisher (more than 50 games)
    # probably drop Screenshots.
    # drop AppID
    # drop Name

    # print("Columns", df.columns)

    df.drop(columns=['Header image', 'Score rank', 'Developers', 'Publishers', 'Screenshots', 'AppID', 'Name'], inplace=True)

    # TRANSFORM
    # transform Website to has_website
    df['has_website'] = df['Website'].notnull().astype(int)
    df.drop(columns=['Website'], inplace=True)

    # tranform Support url to has_support_url
    df['has_support_url'] = df['Support url'].notnull().astype(int)
    df.drop(columns=['Support url'], inplace=True)

    # transform Support email to has_support_email
    df['has_support_email'] = df['Support email'].notnull().astype(int)
    df.drop(columns=['Support email'], inplace=True)

    # transform Metacritic url to has_metacritic_url
    df['has_metacritic_url'] = df['Metacritic url'].notnull().astype(int)
    df.drop(columns=['Metacritic url'], inplace=True)

    def clean_cat_feats(df, feat_name, sep=','):
        all_categories = []
        def clean_categories(categories):
            if type(categories) is not str:
                return []
            return categories.split(sep)

        new_df = df[feat_name].apply(clean_categories)
        # merge all the lists into one
        for categories in new_df:
            all_categories.extend(categories)

        unique_categories = set(all_categories)
        # print(feat_name, len(unique_categories))
        print(f"Unique categories for {feat_name}: {len(unique_categories)}")
        if len(unique_categories) > 500:
            raise ValueError("Too many unique values")
        
        # create a new column for each category
        new_cols = []
        for category in unique_categories:
            new_col = df[feat_name].str.contains(category).astype(int)
            new_cols.append(new_col)
        
        new_df = pd.concat(new_cols, axis=1)
        unique_colums = [feat_name + ' ' + category for category in unique_categories]
        new_df.columns = unique_colums
        # print(new_df.columns)
        # print(new_df.shape)
        df = pd.concat([df, new_df], axis=1)
        # df.merge(new_df, left_index=True, right_index=True, inplace=True)
        
        df.drop(columns=[feat_name], inplace=True)
        return df


    df.dropna(subset=['Categories', 'Genres', 'Tags', 'Movies'], inplace=True)
    # transform Categories (unique vals = 40) using one hot encoding and fill missing values (3407).
    # df = clean_cat_feats(df, 'Categories')
    df.drop(columns=['Categories'], inplace=True)
    df.drop(columns=['Genres'], inplace=True)

    # raise ValueError("Too many unique values")
    # transform Genres (unique vals = 30) using one hot encoding and fill missing values (2439).
    # df = clean_cat_feats(df, 'Genres')
    # transform Tags (unique vals = 446) using one hot encoding and fill missing values (14014). Or maybe not. Just ignore it.
    # df = clean_cat_feats(df, 'Tags')
    df.drop(columns=['Tags'], inplace=True)
    # tranform Movies to num_movies (not sure though. These are NOT actual movies. They are trailers. So, maybe we can ignore this feature.)
    df['num_movies'] = df['Movies'].apply(lambda x: len(x.split(',')))
    df.drop(columns=['Movies'], inplace=True)

    def clean_cat_feats_langs(df, feat_name, sep=','):
        all_categories = []
        def clean_categories(categories):
            if type(categories) is not str:
                return []
            # error in the dataset 
            to_replace = "K'iche'"
            if to_replace in categories:
                categories = categories.replace(to_replace, "'Kiche'")
            try:
                langs = json.loads(categories.replace("'", "\""))
            except:
                raise
            return langs
        
        new_df = df[feat_name].apply(clean_categories)
        # merge all the lists into one
        for categories in new_df:
            all_categories.extend(categories)

        unique_categories = set(all_categories)
        # print(feat_name, len(unique_categories))

        # create a new column for each category
        new_cols = []
        for category in unique_categories:
            try: 
                new_col = df[feat_name].str.contains(category).astype(int)
                new_cols.append(new_col)
            except: 
                print([category])
                raise
        
        new_df = pd.concat(new_cols, axis=1)
        unique_colums = [feat_name + ' ' + category for category in unique_categories]
        new_df.columns = unique_colums
        # print(new_df.columns)
        # print(new_df.shape)
        df = pd.concat([df, new_df], axis=1)
        # df.merge(new_df, left_index=True, right_index=True)

        df.drop(columns=[feat_name], inplace=True)
        return df

    # Supported languages (unique = 134) one hot encoding
    df.drop(columns=['Supported languages'], inplace=True)
    df.drop(columns=['Full audio languages'], inplace=True)

    # df = clean_cat_feats_langs(df, 'Supported languages')
    # Full audio languages (unique = 121) one hot encoding
    # df = clean_cat_feats_langs(df, 'Full audio languages')


    # KEEP
    # Price
    # Required age
    # Release date
    # extract some useful feats from Release date
    df['Release date'] = pd.to_datetime(df['Release date'])
    df['release_year'] = df['Release date'].dt.year
    df['release_month'] = df['Release date'].dt.month
    df['release_day'] = df['Release date'].dt.day
    df.drop(columns=['Release date'], inplace=True)
    # Metacritic score 
    # Achievements
    # Windows 
    # to int
    df['Windows'] = df['Windows'].astype(int)
    # Mac
    df['Mac'] = df['Windows'].astype(int)
    # Linux
    df['Linux'] = df['Windows'].astype(int)


    # ALL BELOW ARE TEXT
    df.drop(columns=['About the game', 'Reviews', 'Notes'], inplace=True)
    # transform About the game to something
    # tranform Reviews to something
    # transform Notes to something

    # TARGET
    df.drop(columns=['Average playtime forever', 'Median playtime forever', 'Median playtime two weeks'], inplace=True)
    # choose one of the ones below and drop the rest to avoid data leakage
    # Average playtime forever          
    # Average playtime two weeks        
    # Median playtime forever           
    # Median playtime two weeks   

    # IDK (but problably drop because leakage)
    # df.dropna(subset=['Average playtime forever', 'Median playtime forever', 'Median playtime two weeks'], inplace=True)
    # Estimated owners
    df['estimated_owner_min'] = df['Estimated owners'].apply(lambda x: int(x.split('-')[0].strip()))
    df['estimated_owner_max'] = df['Estimated owners'].apply(lambda x: int(x.split('-')[1].strip()))
    df.drop(columns=['Estimated owners'], inplace=True)
    # Peak CCU
    # User score
    # Positive
    # Negative
    # Recommendations
    gc.collect()
    if only_x:
        return df
    target_col = 'Average playtime two weeks'
    X = df.drop(columns=[target_col])
    y = df[[target_col]]
    return X, y

def extract_data(project_root):
    df = pd.read_csv(f'{project_root}/data/samples/sample.csv')
    version_file = f'{project_root}/configs/data_version.yaml'
    with open(version_file, 'r') as f:
        version_data = yaml.safe_load(f)
    return df, str(version_data['data_version'])

def load_features(X, y, version):
    print(f"Loading features and target with version {version}")
    zenml.save_artifact(data = pd.concat([X,y], axis=1), name = "features_target", tags=[version])
    return X, y

def validate_transformed_data(X, y):
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
    assert X.isna().sum().sum() == 0, "X should not have missing values"
    assert y.isna().sum().sum() == 0, "y should not have missing values"
    
    cols = X.columns
    types = X.dtypes
    for col, typ in zip(cols, types):
        assert typ == int or typ == float, f"Column {col} should be numeric"
        # assert str(typ).startswith('int') or str(typ).startswith('float')
    assert y[y.columns[0]].dtype == 'int', "y should be numeric"
    return X, y

def validate_app_id(validator: Validator):
    validator.expect_column_values_to_be_unique(column="AppID")
    validator.expect_column_values_to_be_between(
        column="AppID",
        min_value=0,
        # max_value=1_000_000
    )
    validator.expect_column_values_to_not_be_null("AppID")

def validate_release_date(validator: Validator):
    col_name = "Release date"
    # the format is Oct 21, 2008
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}$"
    )
    validator.expect_column_values_to_not_be_null(col_name)
    
def validate_user_score(validator: Validator):
    col_name = "User score"
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=100
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_metacritic_score(validator: Validator):
    col_name = 'Metacritic score'
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=100
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_support_url(validator: Validator):
    col_name = 'Support url'
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=URL_REGEX
    )

def validate_metacritic_url(validator: Validator):
    col_name = 'Metacritic url'
    
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=URL_REGEX
    )

def validate_support_email(validator: Validator):
    col_name = 'Support email'
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=r"^.+@.+\..+$"
    )

def validate_estimated_owners(validator: Validator):
    col_name = 'Estimated owners'
    # use regex to match the pattern
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=r"^\d+ - \d+$"
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_website(validator: Validator):
    col_name = 'Website'
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=URL_REGEX
    )

def validate_peak_ccu(validator: Validator):
    col_name = 'Peak CCU'
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=100_000_000
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_required_age(validator: Validator):
    col_name = 'Required age'
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=100
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_price(validator: Validator):
    col_name = 'Price'
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=1_000
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_dlc_count(validator: Validator):
    col_name = 'DLC count'
    validator.expect_column_values_to_be_between(
        column=col_name,
        min_value=0,
        max_value=100
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_supported_languages(validator: Validator):
    col_name = 'Supported languages'
    # they are in the format: "['English', 'French', ...]"
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=r"^\[(?:(?:\s*\'[^']*\')(?:\s*,\s*\'[^']*\')*\s*)?\]$"
    )
    validator.expect_column_values_to_not_be_null(col_name)

def validate_full_audio_languages(validator: Validator):
    col_name = 'Full audio languages'
    # they are in the format: "['English', 'French', ...]"
    validator.expect_column_values_to_match_regex(
        column=col_name,
        regex=r"^\[(?:(?:\s*\'[^']*\')(?:\s*,\s*\'[^']*\')*\s*)?\]$"
    )
    validator.expect_column_values_to_not_be_null(col_name)


def validate_initial_data(df):
    try:
        context = gx.get_context(context_root_dir = "../services/gx")
    except:
        context = FileDataContext(context_root_dir = "../services")
    
    ds1 = context.sources.add_or_update_pandas(name="my_pandas_ds")
    da1 = ds1.add_dataframe_asset(name="playtime_asset")
    batch_request = da1.build_batch_request(dataframe=df)
    # Read a single data file
    # da1 = ds1.add_csv_asset(
    #     name = "asset01",
    #     filepath_or_buffer="../data/sample.csv")
    
    # Create an Expectation Suite
    expectation_suite_name = "playtime_expectations"
    context.add_or_update_expectation_suite(expectation_suite_name)

    # Create a Validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name
    )
    validator_funcs = [validate_app_id, validate_release_date, validate_user_score, validate_metacritic_score, validate_support_url, validate_metacritic_url, validate_support_email, validate_estimated_owners, validate_website, validate_peak_ccu, validate_required_age, validate_price, validate_dlc_count, validate_supported_languages, validate_full_audio_languages]
    for func in validator_funcs:
        func(validator)

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




config_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'configs'
)

# @hydra.main(config_path=config_path, config_name="config")
def sample_data() -> None:
    # Download the zip file from the URL specified in the config
    # data_url = cfg.dataset.url
    data_url = "https://lime-negative-badger-175.mypinata.cloud/ipfs/QmQDhADFRQmwnNR9sXy2R6YoQbXgLy1G7TFdntBpW64hxg"
    print(f"Downloading data from {data_url}")
    response = requests.get(data_url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download file from {data_url}")

    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))

    # Initialize the progress bar
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=data_url.split('/')[-1]) as pbar:
        buffer = BytesIO()
        for chunk in response.iter_content(1024):
            buffer.write(chunk)
            pbar.update(len(chunk))

    # Extract the zip file
    buffer.seek(0)

    # Extract the zip file
    with ZipFile(buffer) as thezip:
        # List all files in the zip
        zip_info_list = thezip.infolist()
        print("Files in the zip archive:")
        for zip_info in zip_info_list:
            print(zip_info.filename)

        # Extract the specific csv file
        # TODO: Fix
        with thezip.open('games.csv') as thefile:
            df = pd.read_csv(thefile)

    # Sample the data
    # sample_size = cfg.dataset.sample_size
    sample_size = 0.2
    sample_df = df.sample(frac=sample_size, random_state=1)

    # Ensure the sample path exists
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/samples/sample.csv"
        # cfg.dataset.sample_path
    )
    #cfg.dataset.sample_path
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)

    # Save the sample data
    sample_df.to_csv(sample_path, index=False)
    print(f"Sampled data saved to {sample_path}")
    return sample_df


from great_expectations.core.batch import BatchRequest
from great_expectations.data_context import FileDataContext


# @hydra.main(config_path=config_path, config_name="config")
def validate_initial_data():
    # context = get_context()

    context_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "services/gx"
    )

    context = FileDataContext(context_root_dir=context_path)

    ds1 = context.sources.add_or_update_pandas(name="my_pandas_ds")

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/samples/sample.csv"
        # cfg.dataset.sample_path
    )

    da1 = ds1.add_csv_asset(
        name = "sample",
        filepath_or_buffer=sample_path
    )

    suite_name = "sample_suite"

    # Create an expectation suite
    # suite = context.create_expectation_suite(expectation_suite_name=suite_name, overwrite_existing=True)

    context.add_or_update_expectation_suite(suite_name)

    # Define the expectations
    batch_request = da1.build_batch_request()
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    # Example expectations
    # validator.expect_column_values_to_be_between(
    # column="Metacritic score",
    # min_value=0,
    # max_value=100
    # )

    # Example for "User score"
    # validator.expect_column_values_to_be_between(
    #     column="User score",
    #     min_value=0,
    #     max_value=100
    # )

    validator_funcs = [validate_app_id, validate_release_date, validate_user_score, validate_metacritic_score, validate_support_url, validate_metacritic_url, validate_support_email, validate_estimated_owners, validate_website, validate_peak_ccu, validate_required_age, validate_price, validate_dlc_count, validate_supported_languages, validate_full_audio_languages]
    for func in validator_funcs:
        print(func.__name__)
        func(validator)

    # ex3 = validator.expect_column_values_to_be_unique(column = 'AppID', meta = {"dimension": 'Uniqueness'})
    # print(ex3)

    # Save the expectation suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name = "initial_data_validation_checkpoint",
        validations=[
            {
                "batch_request":batch_request,
                "expectation_suite_name" : suite_name
            }
        ]
    )
    
    checkpoint_result = checkpoint.run()

    return checkpoint_result.success


def run_checkpoint():
    context_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "services/gx"
    )

    context = FileDataContext(context_root_dir=context_path)

    checkpoint = context.get_checkpoint("initial_data_validation_checkpoint")

    checkpoint_result = checkpoint.run()

    return checkpoint_result.success


def validate_features(X, y):
    context_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "services/gx"
    )

    context = FileDataContext(context_root_dir=context_path)

    ds1 = context.sources.add_or_update_pandas(name="transformed_ds")
    da_x = ds1.add_dataframe_asset("transformed_features", X)
    # da_y = ds1.add_dataframe_asset("target_data", y)
    # da1 = ds1.add_csv_asset(
    #     name = "sample",
    #     filepath_or_buffer=sample_path
    # )

    suite_name = "transformed_features_validation_suite"

    # Create an expectation suite
    # suite = context.create_expectation_suite(expectation_suite_name=suite_name, overwrite_existing=True)

    context.add_or_update_expectation_suite(suite_name)

    # Define the expectations
    batch_request = da_x.build_batch_request()
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    # validator_funcs = []
    # for func in validator_funcs:
    #     print(func.__name__)
    #     func(validator)


    # Check that X and y have the same number of rows
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"

    # Check that all columns in X are either int or float using great_expectations
    for col in X.columns:
        if col != "Price":
            validator.expect_column_values_to_be_of_type(column=col, type_="int64")
    # Price should be float64
    validator.expect_column_values_to_be_of_type(column="Price", type_="float64")


    binary_feats_prefix = ["Windows", "Mac", "Linux", "has_website", 'has_support_url', 'has_support_email', 'has_metacritic_url'
                       "Categories ", "Supported languages ", "Full audio languages ", "Tags ", "Genres ", "Categories "]
    for col in X.columns:
        for prefix in binary_feats_prefix:
            if col.startswith(prefix):
                validator.expect_column_distinct_values_to_be_in_set(column=col, value_set=[0, 1])


    # Save the expectation suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name = "transformed_data_validation_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name" : suite_name
            }
        ]
    )
    
    checkpoint_result = checkpoint.run()

    if not checkpoint_result.success:
        raise Exception("Checkpoint failed!")

    return X, y


if __name__ == "__main__":
    # sample_data()
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/samples/sample.csv"
    )
    df = pd.read_csv(sample_path)
    validate_initial_data()
    result = run_checkpoint()



