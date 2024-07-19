import pytest
import os
import requests
import pandas as pd
from omegaconf import DictConfig
from unittest import mock
from io import BytesIO
from zipfile import ZipFile
from src.data import sample_data  # assuming the function is in a file named sample_script.py

# Sample configuration for testing
sample_cfg = DictConfig({
    'dataset': {
        'url': 'http://example.com/data.zip',
        'sample_size': 0.1,
        'sample_path': 'data/sample.csv'
    }
})

# Mock data for testing
mock_csv_data = "column1,column2\nvalue1,value2\nvalue3,value4"
mock_zip_data = BytesIO()
with ZipFile(mock_zip_data, 'w') as zf:
    zf.writestr('games.csv', mock_csv_data)
mock_zip_data.seek(0)

@pytest.fixture
def mock_requests_get(mocker):
    return mocker.patch('requests.get')

def test_download_failure(mock_requests_get):
    mock_requests_get.return_value.status_code = 404

    with pytest.raises(Exception) as excinfo:
        sample_data(sample_cfg)
    
    assert "Failed to download file" in str(excinfo.value)

def test_successful_data_sampling(mocker, mock_requests_get, tmpdir):
    # Mock the successful request
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = mock_zip_data.getvalue()

    # Mock os.path.dirname and os.makedirs
    mocker.patch('os.path.dirname', return_value=str(tmpdir))
    mocker.patch('os.makedirs')

    # Mock the path to save the sampled data
    sample_cfg.dataset.sample_path = os.path.join(tmpdir, 'sample.csv')

    # Run the function
    sample_data(sample_cfg)

    # Verify that the data was saved correctly
    saved_sample_path = sample_cfg.dataset.sample_path
    assert os.path.exists(saved_sample_path)

    # Verify the content of the saved file
    saved_df = pd.read_csv(saved_sample_path)
    assert not saved_df.empty
    assert list(saved_df.columns) == ["column1", "column2"]

