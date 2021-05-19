import pytest
import pandas as pd
import numpy as np
from faker import Faker
from ml_example.data.make_dataset import read_data, split_train_val_data
from ml_example.enities import SplittingParams


def generate_test_dataset(size_dataset: int) -> pd.DataFrame:
    test_data = Faker()
    age = np.random.randint(29, 77, size=size_dataset, dtype='int')
    sex = np.random.randint(0, 2, size=size_dataset, dtype='int')
    cp = np.random.randint(0, 4, size=size_dataset, dtype='int')
    trestbps = np.random.randint(94, 201, size=size_dataset, dtype='int')
    chol = np.random.randint(126, 565, size=size_dataset, dtype='int')
    fbs = test_data.random_elements(elements=OrderedDict([(0, 0.68), (1, 0.32), ]),
                                    unique=False, length=size_dataset)
    restecg = test_data.random_elements(elements=OrderedDict([(0, 0.48), (1, 0.48), (2, 0.04), ]),
                                        unique=False, length=size_dataset
                                        )
    thalach = np.random.randint(71, 203, size=size_dataset, dtype='int')
    exang = test_data.random_elements(elements=OrderedDict([(0, 0.68), (1, 0.32), ]),
                                      unique=False, length=size_dataset)
    oldpeak = []
    for _ in range(size_dataset):
        oldpeak.append(round(np.random.uniform(0., 6.2), 1))
    slope = test_data.random_elements(elements=OrderedDict([(0, 0.1), (1, 0.45), (2, 0.45), ]),
                                      unique=False, length=size_dataset)

    ca = test_data.random_elements(elements=OrderedDict([(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05), ]),
                                   unique=False, length=size_dataset)

    thal = test_data.random_elements(elements=OrderedDict([(0, 0.1), (1, 0.2), (2, 0.4), (3, 0.3), ]),
                                     unique=False, length=size_dataset)

    target = np.random.randint(0, 2, size=size_dataset, dtype='int')

    data = pd.DataFrame(list(zip(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                 exang, oldpeak, slope, ca, thal, target)),
                        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
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
    splitting_params = SplittingParams(random_state=42, val_size=val_size, )
    data = read_data_fixture
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10


def test_generate_fake_dataset(size_dataset: int):
    original_data = pd.read_csv('data/raw/heart.csv')
    data = np.random.choice(original_data.index.values, size_dataset)
    assert size_dataset == data.shape[0]
