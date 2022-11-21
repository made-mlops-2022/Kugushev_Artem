"""
File which runs all train pipline
"""

import json
import os
from typing import Dict, Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from data.dataclass.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    create_inference_pipeline,
    serialize_model
)

from logs.create_logger import create_logger


def train_pipeline(config_path: str) -> Tuple[str, Dict[str, float]]:
    """
    Get training params and run pipeline
    :param config_path: path to yaml config file, parent dir is ./configs
    :return: results of running training pipline
    """

    training_pipeline_params = read_training_pipeline_params(config_path)

    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params: TrainingPipelineParams) -> Tuple[str, Dict[str, float]]:
    """
    Full train pipeline
    :param training_pipeline_params: params for pipeline training
    :return: saved model path and evaluation metrics
    """

    logger = create_logger(training_pipeline_params.log_config_file_path, training_pipeline_params.log_format)

    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    data = pd.read_csv(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    splitting_params = training_pipeline_params.splitting_params

    train_df, val_df = train_test_split(
        data, test_size=splitting_params.val_size, random_state=splitting_params.random_state)

    train_features, train_target = split_x_y(train_df, training_pipeline_params.feature_params.target_name)
    val_features, val_target = split_x_y(val_df, training_pipeline_params.feature_params.target_name)

    logger.info(f"train_df.shape is {train_features.shape}")
    logger.info(f"val_df.shape is {val_features.shape}")

    model = train_model(train_features, train_target, training_pipeline_params.train_params.model_type)
    inference_pipeline = create_inference_pipeline(model)

    train_predicts = predict_model(inference_pipeline, train_features)
    train_metrics = evaluate_model(train_predicts, train_target)
    logger.info(f"Train metrics: {train_metrics}")

    predicts = predict_model(inference_pipeline, val_features)
    metrics = evaluate_model(predicts, val_target)

    name_of_new_dir = "/".join(training_pipeline_params.metric_path.split("/")[:-1])
    os.makedirs(name_of_new_dir, exist_ok=True)
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Validation metrics: {metrics}")

    path_to_model = serialize_model(inference_pipeline, training_pipeline_params.output_model_path)

    logger.info(f"Finish train {training_pipeline_params.train_params.model_type}")
    return path_to_model, metrics


def split_x_y(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset to X, y parts
    :param df: full dataframe
    :param target_name: name of target column
    :return:
    """

    features = df.drop(target_name, axis=1)
    target = df[target_name]
    return features, target


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
