import os
import pandas as pd
import hydra
from omegaconf import DictConfig
import requests
from zipfile import ZipFile
from io import BytesIO

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@hydra.main(config_path="../configs", config_name="config")
def sample_data(cfg: DictConfig) -> None:
    # Download the zip file from the URL specified in the config
    data_url = cfg.dataset.url
    response = requests.get(data_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file from {data_url}")

    # Extract the zip file
    with ZipFile(BytesIO(response.content)) as thezip:
        # List all files in the zip
        zip_info_list = thezip.infolist()
        print("Files in the zip archive:")
        for zip_info in zip_info_list:
            print(zip_info.filename)

        # Extract the specific csv file
        with thezip.open('games.csv') as thefile:
            df = pd.read_csv(thefile)

    # Sample the data
    sample_size = cfg.dataset.sample_size
    sample_df = df.sample(frac=sample_size, random_state=1)

    # Ensure the sample path exists
    sample_path = cfg.dataset.sample_path
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)

    # Save the sample data
    sample_df.to_csv(sample_path, index=False)
    print(f"Sampled data saved to {sample_path}")

if __name__ == "__main__":
    sample_data()
