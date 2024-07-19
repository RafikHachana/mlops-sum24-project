
import pandas as pd
import json
import gc
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

def transform_data(df):
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
        print(feat_name, len(unique_categories))
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
    df = clean_cat_feats(df, 'Categories')
    # raise ValueError("Too many unique values")
    # transform Genres (unique vals = 30) using one hot encoding and fill missing values (2439).
    df = clean_cat_feats(df, 'Genres')
    # transform Tags (unique vals = 446) using one hot encoding and fill missing values (14014). Or maybe not. Just ignore it.
    df = clean_cat_feats(df, 'Tags')
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
        print(feat_name, len(unique_categories))

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
    df = clean_cat_feats_langs(df, 'Supported languages')
    # Full audio languages (unique = 121) one hot encoding
    df = clean_cat_feats_langs(df, 'Full audio languages')


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

    target_col = 'Average playtime two weeks'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def extract_data(BASE_PATH):
    df = pd.read_csv(BASE_PATH)
    version = "v1.0"
    return df, version

def load_features(X, y, version):
    print(f"Loading features and target with version {version}")

def validate_transformed_data(X, y):
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
    assert X.isna().sum().sum() == 0, "X should not have missing values"
    assert y.isna().sum().sum() == 0, "y should not have missing values"
    
    cols = X.columns
    types = X.dtypes
    for col, typ in zip(cols, types):
        assert typ == int or typ == float, f"Column {col} should be numeric"
        # assert str(typ).startswith('int') or str(typ).startswith('float')
    assert y.dtype == 'int', "y should be numeric"
    return X, y


config_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'configs'
)

# @hydra.main(config_path=config_path, config_name="config")
def sample_data() -> None:
    # Download the zip file from the URL specified in the config
    # data_url = cfg.dataset.url
    data_url = "https://drive.usercontent.google.com/u/0/uc?id=1aw-Xu5T4UW6fr7icR7o30zPlagii9IV4&export=download"
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
        with thezip.open('small.csv') as thefile:
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

    ex3 = validator.expect_column_values_to_be_unique(column = 'AppID', meta = {"dimension": 'Uniqueness'})
    print(ex3)

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


if __name__ == "__main__":
    # sample_data()
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/samples/sample.csv"
    )
    df = pd.read_csv(sample_path)
    validate_initial_data()
    result = run_checkpoint()

