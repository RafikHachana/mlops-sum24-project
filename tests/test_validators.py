
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pandas as pd
from great_expectations.dataset import PandasDataset
from src.data import (
    validate_app_id,
    validate_release_date,
    validate_user_score,
    validate_metacritic_score,
    validate_support_url,
    validate_metacritic_url,
    validate_peak_ccu,
    validate_required_age,
    validate_price,
    validate_dlc_count,
    validate_supported_languages,
    validate_full_audio_languages,
    validate_estimated_owners,
    validate_website
)

@pytest.fixture
def success_validator():
    df = pd.DataFrame({
        "AppID": [1, 2, 3],
        "Release date": ["Oct 21, 2008", "Nov 05, 2010", "Dec 31, 2021"],
        "User score": [85, 90, 70],
        "Metacritic score": [88, 91, 85],
        "Support url": ["http://support.com", "https://support.org", "http://help.net"],
        "Metacritic url": ["http://metacritic.com/game1", "https://metacritic.org/game2", "http://metacritic.net/game3"],
        "Peak CCU": [500, 2000, 15000],
        "Required age": [0, 12, 18],
        "Price": [0, 49.99, 59.99],
        "DLC count": [0, 2, 5],
        "Supported languages": ["['English', 'French']", "['German', 'Spanish']", "['Chinese', 'Japanese']"],
        "Full audio languages": ["['English', 'French']", "['German', 'Spanish']", "['Chinese', 'Japanese']"],
        "Estimated owners": ["1000 - 5000", "500 - 2000", "10000 - 50000"],
        "Website": ["http://example.com", "https://example.org", "http://example.net"]
    })
    return PandasDataset(df)

@pytest.fixture
def fail_validator():
    df = pd.DataFrame({
        "AppID": [1, 1, 3],  # Duplicate AppID
        "Release date": ["October 21, 2008", "11/05/2010", "2021-12-31"],  # Incorrect date format
        "User score": [85, 150, 70],  # Score out of range
        "Metacritic score": [88, -5, 105],  # Score out of range

        "Support url": [2, 2.3, 5],  # Incorrect URL format
        "Metacritic url": [5, 10, -5],  # Incorrect URL format

        "Peak CCU": [500, -2000, 1500000000],  # Out of range
        "Required age": [-1, 101, 18],  # Out of range
        "Price": [-10, 1500, 59.99],  # Out of range
        "DLC count": [-1, 200, 5],  # Out of range
        "Supported languages": ["English, French", "['German', Spanish]", "['Chinese', 'Japanese']"],  # Incorrect format
        "Full audio languages": ["['English', French']", "German, 'Spanish'", "['Chinese', 'Japanese']"],  # Incorrect format
        "Estimated owners": ["1000_5000", "500-2000", "10000to50000"],  # Incorrect formats
        "Website": [-1, 3.5, 100] # Incorrect formats
    })
    return PandasDataset(df)

@pytest.fixture
def null_validator():
    df = pd.DataFrame({
        "AppID": [None, None, None],
        "Release date": [None, None, None],
        "User score": [None, None, None],
        "Metacritic score": [None, None, None],
        "Support url": [None, None, None],
        "Metacritic url": [None, None, None],
        "Peak CCU": [None, None, None],
        "Required age": [None, None, None],
        "Price": [None, None, None],
        "DLC count": [None, None, None],
        "Supported languages": [None, None, None],
        "Full audio languages": [None, None, None],
        "Estimated owners": [None, None, None],
        "Website": [None, None, None]
    })
    return PandasDataset(df)

def test_validate_app_id_success(success_validator):
    validate_app_id(success_validator)
    assert success_validator.validate().success

def test_validate_app_id_failure_unique(fail_validator):
    validate_app_id(fail_validator)
    assert not fail_validator.validate().success

def test_validate_app_id_failure_null(null_validator):
    validate_app_id(null_validator)
    assert not null_validator.validate().success

def test_validate_release_date_success(success_validator):
    validate_release_date(success_validator)
    assert success_validator.validate().success

def test_validate_release_date_failure_invalid_format(fail_validator):
    validate_release_date(fail_validator)
    assert not fail_validator.validate().success

def test_validate_release_date_failure_null(null_validator):
    validate_release_date(null_validator)
    assert not null_validator.validate().success

def test_validate_user_score_success(success_validator):
    validate_user_score(success_validator)
    assert success_validator.validate().success

def test_validate_user_score_failure_null(null_validator):
    validate_user_score(null_validator)
    assert not null_validator.validate().success

def test_validate_user_score_failure_out_of_range(fail_validator):
    validate_user_score(fail_validator)
    assert not fail_validator.validate().success

def test_validate_metacritic_score_success(success_validator):
    validate_metacritic_score(success_validator)
    assert success_validator.validate().success

def test_validate_metacritic_score_failure_out_of_range(fail_validator):
    validate_metacritic_score(fail_validator)
    assert not fail_validator.validate().success

def test_validate_metacritic_score_failure_null(null_validator):
    validate_metacritic_score(null_validator)
    assert not null_validator.validate().success

def test_validate_support_url_success(success_validator):
    validate_support_url(success_validator)
    assert success_validator.validate().success

def test_validate_support_url_failure(fail_validator):
    validate_support_url(fail_validator)
    assert not fail_validator.validate().success

def test_validate_metacritic_url_success(success_validator):
    validate_metacritic_url(success_validator)
    assert success_validator.validate().success

def test_validate_metacritic_url_failure(fail_validator):
    validate_metacritic_url(fail_validator)
    assert not fail_validator.validate().success

def test_validate_peak_ccu_success(success_validator):
    validate_peak_ccu(success_validator)
    assert success_validator.validate().success

def test_validate_peak_ccu_failure(fail_validator):
    validate_peak_ccu(fail_validator)
    assert not fail_validator.validate().success

def test_validate_required_age_success(success_validator):
    validate_required_age(success_validator)
    assert success_validator.validate().success

def test_validate_required_age_failure(fail_validator):
    validate_required_age(fail_validator)
    assert not fail_validator.validate().success

def test_validate_price_success(success_validator):
    validate_price(success_validator)
    assert success_validator.validate().success

def test_validate_price_failure(fail_validator):
    validate_price(fail_validator)
    assert not fail_validator.validate().success

def test_validate_dlc_count_success(success_validator):
    validate_dlc_count(success_validator)
    assert success_validator.validate().success

def test_validate_dlc_count_failure(fail_validator):
    validate_dlc_count(fail_validator)
    assert not fail_validator.validate().success

def test_validate_supported_languages_success(success_validator):
    validate_supported_languages(success_validator)
    assert success_validator.validate().success

def test_validate_supported_languages_failure(fail_validator):
    validate_supported_languages(fail_validator)
    assert not fail_validator.validate().success

def test_validate_full_audio_languages_success(success_validator):
    validate_full_audio_languages(success_validator)
    assert success_validator.validate().success

def test_validate_full_audio_languages_failure(fail_validator):
    validate_full_audio_languages(fail_validator)
    assert not fail_validator.validate().success

def test_validate_estimated_owners_success(success_validator):
    validate_estimated_owners(success_validator)
    assert success_validator.validate().success

def test_validate_estimated_owners_failure(fail_validator):
    validate_estimated_owners(fail_validator)
    assert not fail_validator.validate().success

def test_validate_website_success(success_validator):
    validate_website(success_validator)
    assert success_validator.validate().success

def test_validate_website_failure(fail_validator):
    validate_website(fail_validator)
    assert not fail_validator.validate().success

