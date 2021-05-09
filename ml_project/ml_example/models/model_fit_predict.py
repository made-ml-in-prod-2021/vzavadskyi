import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    plot_confusion_matrix,
    plot_roc_curve,
)

from ml_example.enities.train_params import TrainingParams
from ml_example.enities.train_pipeline_params import TrainingPipelineParams

SklearnModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
        features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnModel:
    if train_params.model_type == 'KNeighborsClassifier':
        model = KNeighborsClassifier(leaf_size=train_params.leaf_size,
                                     n_neighbors=train_params.n_neighbors, p=train_params.p,
                                     weights=train_params.weights, n_jobs=-1
                                     )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: SklearnModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "auc": roc_auc_score(target, predicts),
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts)
    }


def report_model(model: SklearnModel, features: pd.DataFrame, target: np.ndarray, train_params: TrainingParams,
                 report_path: TrainingPipelineParams) -> str:
    report_file_path = f'{report_path}/{train_params.model_type}_metrics.png'
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_confusion_matrix(model, features, target, ax=axes[0])
    plot_roc_curve(model, features, target, ax=axes[1])
    plt.title(train_params.model_type)
    plt.savefig(report_file_path)
    return report_file_path


def serialize_model(model: SklearnModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
