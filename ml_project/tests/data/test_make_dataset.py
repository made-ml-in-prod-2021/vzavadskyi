import pytest

from ml_example.data.make_dataset import read_data, split_train_val_data
from ml_example.enities import SplittingParams
import pandas as pd
import numpy as np



def generate_test_dataset(size_dataset: int) -> pd.DataFrame:
    df = pd.read_csv('data/raw/heart.csv')
    rows = np.random.choice(df.index.values, size_dataset)

    data = df.loc[rows]
    return data


@pytest.fixture(scope='session')
def read_data_fixture(dataset_path: str):
    data = read_data(dataset_path)
    return data


def test_read_data(read_data_fixture: pd.DataFrame, target_col: str):
    data = read_data_fixture
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_train_val_data(read_data_fixture: pd.DataFrame):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=42, val_size=val_size,)
    data = read_data_fixture
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10


def test_generate_fake_dataset(size_dataset: int):
    original_data = pd.read_csv('data/raw/heart.csv')
    data = np.random.choice(original_data.index.values, size_dataset)
    assert size_dataset == data.shape[0]

