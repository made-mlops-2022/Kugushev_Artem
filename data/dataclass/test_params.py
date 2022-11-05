"""
Class providing train-test split params for training pipeline
"""

from dataclasses import dataclass, field

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class TestParams:
    log_config_file_path: str = field(default="configs/logger_config.yaml")
    log_format: str = field(default="log_to_file")


TestParamsSchema = class_schema(TestParams)


def read_test_params(path: str) -> TestParams:
    """
    Load test params from config file
    :param path: path to config file
    :return: schema of test params
    """

    with open(path, "r") as input_stream:
        schema = TestParamsSchema()
        return schema.load(yaml.safe_load(input_stream))