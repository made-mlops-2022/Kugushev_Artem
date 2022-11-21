"""
Class providing params for training pipeline
"""


from dataclasses import dataclass

from data.dataclass.feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    predict_path: str
    model_type: str
    feature_params: FeatureParams
    log_config_file_path: str
    log_format: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    """
    Load predict params from config file
    :param path: path to config file
    :return: schema of training params
    """

    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
