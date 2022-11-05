"""
Class providing params for training pipeline
"""


from dataclasses import dataclass

from data.dataclass.split_params import SplittingParams
from data.dataclass.train_params import TrainingParams
from data.dataclass.feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams
    log_config_file_path: str
    log_format: str


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    Load training params from config file
    :param path: path to config file
    :return: schema of training params
    """

    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
