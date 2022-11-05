"""
File which runs all train pipline
"""


import logging
import os
import pickle

import click
import numpy as np
import pandas as pd

from data.dataclass.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)
from logs.create_logger import create_logger

from models.model_fit_predict import (
    predict_model,
    create_inference_pipeline
)


def predict_pipeline(config_path: str) -> str:
    """
    Get predict params and run pipeline
    :param config_path: path to yaml config file, parent dir is ./configs
    :return: predictions
    """

    predict_pipeline_params = read_predict_pipeline_params(config_path)

    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> str:
    """
    Full predict pipeline
    :param predict_pipeline_params: params for pipeline training
    :return: saved predictions path
    """

    logger = create_logger(predict_pipeline_params.log_config_file_path, predict_pipeline_params.log_format)

    logger.info(f"Start predict pipeline with params {predict_pipeline_params}")
    data = pd.read_csv(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    if predict_pipeline_params.feature_params.target_name:
        data = data.drop(predict_pipeline_params.feature_params.target_name, axis=1)

    model_type = predict_pipeline_params.model_type
    model_path = get_model_path(model_type, logger)

    with open(model_path, "rb") as model:
        model = pickle.load(model)

    inference_pipeline = create_inference_pipeline(model)

    predicts = predict_model(inference_pipeline, data)

    name_of_new_dir = "/".join(predict_pipeline_params.predict_path.split("/")[:-1])
    os.makedirs(name_of_new_dir, exist_ok=True)
    np.savetxt(predict_pipeline_params.predict_path, predicts, delimiter=",", fmt="%d")

    logger.info(f"Finish predict {predict_pipeline_params.model_type}")

    return predict_pipeline_params.predict_path


def get_model_path(model_type: str, logger: logging.Logger) -> str:
    if model_type == "DecisionTree":
        model_path = "models/tree/model.pkl"
    elif model_type == "LogisticRegression":
        model_path = "models/log_reg/model.pkl"
    else:
        logger.error(f"No such pretrained model: {model_type}")
        raise FileNotFoundError
    return model_path


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
