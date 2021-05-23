import os
from typing import List

from py._path.local import LocalPath

from ml_example.main import train_pipeline
from ml_example.enities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_full_pipeline(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_report_path = tmpdir.join()
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        report_path=expected_report_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=42),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
        ),
        train_params=TrainingParams(model_type="KNeighborsClassifier"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["auc"] > 0.5
