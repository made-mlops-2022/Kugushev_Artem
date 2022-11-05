"""
Class providing tests for predict_pipeline
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


class PredictPipelineTest(TestCase):
    """
    Test functions for predict pipeline
    """

    def test_model_load(self):
        """
        Assert that model path is correct
        :return: None
        """

        logger.info("Start test_model_load function")
        self.assertEqual(get_model_path("DecisionTree", logger), "models/tree/model.pkl")
        self.assertEqual(get_model_path("LogisticRegression", logger), "models/log_reg/model.pkl")
        with self.assertRaises(FileNotFoundError):
            get_model_path("CatBoost", logger)
        logger.info("Finish test_model_load function")

    def test_saving_predictions(self):
        """
        Assert that file created
        :return: None
        """

        logger.info("Start test_saving_predictions function")

        config_path = "configs/test_predict_config.yaml"
        test_predicts = np.array([0, 1, 1, 0])

        predict_pipeline_params = read_predict_pipeline_params(config_path)
        save_preds_to_file(test_predicts, predict_pipeline_params)
        self.assertTrue(os.path.exists("models/predictions/test_predict.csv"))

        results = []
        with open("models/predictions/test_predict.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                results.extend(row)
        array = [int(el) for el in results]
        self.assertTrue((np.array(array) == test_predicts).all())
        logger.info("Finish test_saving_predictions function")
