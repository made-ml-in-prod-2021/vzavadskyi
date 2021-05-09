import json
import logging
import os

import pandas as pd
import hydra
from omegaconf import DictConfig

from ml_example.data import read_data, split_train_val_data
from ml_example.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_example.features import make_features
from ml_example.features.build_features import extract_target, build_transformer
from ml_example.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    report_model,
)

APPLICATION_NAME = 'ml_project'
logger = logging.getLogger(APPLICATION_NAME)


def prepare_val_features_for_predict(
        train_features: pd.DataFrame, val_features: pd.DataFrame
):
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"rows in dataset:{data.shape[0]}")
    logger.info(f"Columns in dataset: {data.shape[1]}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape:{train_df.shape}")
    logger.info(f"val_df.shape:{val_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"train_features.shape:{train_features.shape},  train_target.shape:{train_target.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    logger.info("model created.")

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )
    logger.info(f"val_features_prepared.shape:{val_features_prepared.shape}")

    predicts = predict_model(
        model,
        val_features_prepared,
    )
    logger.info(f"predicts.shape:{predicts.shape}")

    metrics = evaluate_model(
        predicts,
        val_target.to_numpy(),
    )

    report_file_path = report_model(
        model, val_features_prepared, val_target.to_numpy(), training_pipeline_params.train_params,
        training_pipeline_params.report_path
    )
    logger.info(f"report_file_path:{report_file_path}")

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics:{metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    return path_to_model, metrics


@hydra.main(config_name="train_config_1.yaml")
def train_pipeline_command(cfg: DictConfig):
    logger.info("Working directory : {}".format(os.getcwd()))
    params = read_training_pipeline_params(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
