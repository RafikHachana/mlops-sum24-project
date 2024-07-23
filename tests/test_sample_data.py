import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from io import BytesIO
import requests
from zipfile import ZipFile
from omegaconf import DictConfig
from tqdm import tqdm
import src.data

from src.data import sample_data

@patch('src.data.requests.get')
@patch('src.data.tqdm')
@patch('src.data.ZipFile')
@patch('src.data.os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_sample_data(mock_open, mock_makedirs, mock_zipfile, mock_tqdm, mock_get):
    # Prepare mock response for requests.get
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '100'}
    mock_response.iter_content = lambda chunk_size: [b'x' * chunk_size for _ in range(10)]
    mock_get.return_value = mock_response

    # Prepare mock ZipFile
    mock_zip = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    mock_zip.infolist.return_value = [MagicMock(filename='games.csv')]
    mock_zip.open.return_value = BytesIO(b"column1,column2\nvalue1,value2")

    # Mock configuration
    cfg = DictConfig({
        'dataset': {
            'url': 'http://example.com/data.zip',
            'sample_size': 0.5,
            'sample_path': 'data/sample.csv'
        }
    })

    # Run the function
    with patch('pandas.read_csv', return_value=pd.DataFrame({'column1': ['value1'], 'column2': ['value2']})):
        sample_data()

    # Assertions
    # mock_get.assert_called_once_with('http://example.com/data.zip', stream=True)
    mock_tqdm.assert_called_once()
    sample_path = os.path.join(os.path.dirname(os.path.dirname(src.data.__file__)), cfg['dataset']['sample_path'])
    # mock_makedirs.assert_called_once_with(os.path.dirname(sample_path), exist_ok=True)
    assert mock_zip.open.called
    assert mock_zip.infolist.called

# @patch('src.data.requests.get')
# def test_download_failure(mock_get):
#     # Prepare mock response for requests.get
#     mock_response = MagicMock()
#     mock_response.status_code = 404
#     mock_get.return_value = mock_response

#     # Mock configuration
#     cfg = DictConfig({
#         'dataset': {
#             'url': 'http://example.com/data.zip',
#             'sample_size': 0.5,
#             'sample_path': 'data/sample.csv'
#         }
#     })

#     # Run the function and expect an exception
#     with pytest.raises(Exception) as excinfo:
#         sample_data(cfg)

#     assert str(excinfo.value) == 'Failed to download file from http://example.com/data.zip'
#     mock_get.assert_called_once_with('http://example.com/data.zip', stream=True)
