"""
Class providing tests for train_pipeline
"""
import csv
import os
from unittest import TestCase

import numpy as np

from data.dataclass.predict_pipeline_params import read_predict_pipeline_params
from data.dataclass.test_params import read_test_params
from predict_pipeline import get_model_path, save_preds_to_file
from logs.create_logger import create_logger

PATH_TO_CONFIG = "configs/test_config.yaml"

test_params = read_test_params(PATH_TO_CONFIG)
logger = create_logger(test_params.log_config_file_path, test_params.log_format)


class TrainPipelineTest(TestCase):
    """
    Test functions for train pipeline
    """

    def test_split_x_y(self):
        """
        Assert that model path is correct
        :return: None
        """
        pass

    def test_create_pipeline(self):
        """
        Assert that pipeline created
        :return: None
        """
        pass
