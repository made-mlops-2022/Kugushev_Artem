"""
Class providing tests for train_pipeline
"""


from unittest import TestCase

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.dataclass.test_params import read_test_params
from logs.create_logger import create_logger
from models import create_inference_pipeline
from train_pipeline import split_x_y

PATH_TO_CONFIG = "configs/test_config.yaml"

test_params = read_test_params(PATH_TO_CONFIG)
logger = create_logger(test_params.log_config_file_path, test_params.log_format)


class TrainPipelineTest(TestCase):
    """
    Test functions for train pipeline
    """

    def test_split_x_y(self):
        """
        Assert that data splits to X and Y correctly
        :return: None
        """

        logger.info("Start testing train pipeline")
        test_df = pd.DataFrame(data={"feature": [23, 45, 11], "age": [45, 21, 34], "condition": [1, 0, 1]})
        test_features, test_target = split_x_y(test_df, "condition")
        self.assertEqual(test_features.shape, (3, 2))
        self.assertEqual(test_target.shape, (3, ))
        self.assertEqual(test_target.array, [1, 0, 1])
        logger.info("Finish testing train pipeline")


    def test_create_pipeline(self):
        """
        Assert that pipeline created
        :return: None
        """

        logger.info("Start test pipeline creation")
        model = LogisticRegression()
        pipeline = create_inference_pipeline(model)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(pipeline.steps, [("model_part",  model)])
        logger.info("Finish test pipeline creation")
