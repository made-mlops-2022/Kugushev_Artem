"""
File provides functions for logging
"""
import logging.config

import yaml


def create_logger(logger_config_path: str, logger_format: str) -> logging.Logger:
    """
    Function create logger with file handler
    :param logger_format: file or stream handler
    :param logger_config_path: path to output file, given in predict_config.yaml
    :return: created logger
    """

    with open(logger_config_path, 'r') as file:
        config = yaml.safe_load(file)

    logging.config.dictConfig(config)
    logger = logging.getLogger(logger_format)

    return logger
