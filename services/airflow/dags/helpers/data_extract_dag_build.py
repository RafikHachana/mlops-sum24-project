import pandas as pd

def extract_sample():
    # Code to extract a new sample of the data
    data = pd.read_csv("data/games.csv")
    samples = data.sample(5)
    return samples

def validate_sample(data):
    # Code to validate the sample using Great Expectations
    if data.isnull().values.any():
        raise ValueError("Data contains null values")
    return data

def version_sample(data):
    # Code to version the sample using DVC
    version_info = {"version": "v1.0", "data": data}
    return version_info

def load_sample(version_info):
    # Code to load the sample to the data store
    data = version_info["data"]
    print(f"Loading {data} with version {version_info['version']}")