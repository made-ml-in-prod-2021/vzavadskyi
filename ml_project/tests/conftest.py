import os
import tmpdir
import pytest
from typing import List
from ml_example.enities.feature_params import FeatureParams
from tests.data.test_make_dataset import generate_test_dataset

size_of_sample = 1000
sample_dataset_name = "sample_for_test.csv"


@pytest.fixture(scope='session')
def dataset_path():
    data = generate_test_dataset(size_of_sample)
    data.to_csv(f"tests/{sample_dataset_name}")
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, sample_dataset_name)


@pytest.fixture(scope='session')
def target_col():
    return "target"


@pytest.fixture(scope='session')
def size_dataset():
    return size_of_sample


@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]


@pytest.fixture(scope='session')
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params
