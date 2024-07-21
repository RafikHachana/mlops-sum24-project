
import pytest
import pandas as pd
from great_expectations.dataset import PandasDataset
from data import (
    validate_app_id,
    validate_release_date,
    validate_user_score,
    validate_metacritic_score,
    validate_support_url,
    validate_metacritic_url,
)

@pytest.fixture
def success_validator():
    df = pd.DataFrame({
        "AppID": [1, 2, 3],
        "Release date": ["Oct 21, 2008", "Nov 05, 2010", "Dec 31, 2021"],
        "User score": [85, 90, 70],
        "Metacritic score": [88, 91, 85],
        "Support url": ["http://support.com", "https://support.org", "http://help.net"],
        "Metacritic url": ["http://metacritic.com/game1", "https://metacritic.org/game2", "http://metacritic.net/game3"]
    })
    return PandasDataset(df)

@pytest.fixture
def fail_validator():
    df = pd.DataFrame({
        "AppID": [1, 1, 3],  # Duplicate AppID
        "Release date": ["October 21, 2008", "11/05/2010", "2021-12-31"],  # Incorrect date format
        "User score": [85, 150, 70],  # Score out of range
        "Metacritic score": [88, -5, 105],  # Score out of range
        "Support url": ["ftp://support.com", "https://support.org", "htp://help.net"],  # Incorrect URL format
        "Metacritic url": ["http://metacritic.com/game1", "metacritic.org/game2", "http://metacritic.net/game3"]  # Incorrect URL format
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
        "Metacritic url": [None, None, None]
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
