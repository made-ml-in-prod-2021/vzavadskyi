from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ml_example.data.make_dataset import read_data
from ml_example.enities import FeatureParams
from ml_example.features.build_features import (
    make_features, extract_target, build_transformer, process_categorical_features
)


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_exctract_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)
    target = extract_target(data, feature_params)
    assert_allclose(
        data[feature_params.target_col].to_numpy(), target.to_numpy()
    )


@pytest.fixture(scope='session')
def categorical_feature() -> str:
    return "categorical_feature"


@pytest.fixture(scope='session')
def categorical_values() -> List[str]:
    return ["first", "second", "third"]


@pytest.fixture(scope='session')
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture(scope='session')
def fake_categorical_data(
    categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})


def test_process_categorical_features(
    fake_categorical_data: pd.DataFrame
):
    transformed: pd.DataFrame = process_categorical_features(fake_categorical_data)
    assert transformed.shape[1] == 4
    assert transformed.sum().sum() == 4