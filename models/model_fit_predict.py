"""
File with fit, predict and evaluate functions
"""
import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

classificationModels = Union[LogisticRegression, DecisionTreeClassifier]
DECISION_TREE = "DecisionTree"
LOGISTIC_REGRESSION = "LogisticRegression"


def train_model(features: pd.DataFrame, target: pd.Series, model_type: str) -> LogisticRegression:
    """
    Train model
    :param model_type: type of model choosing when write config path
    :param features: train features
    :param target: train column with binary target
    :return: trained model
    """

    if model_type == LOGISTIC_REGRESSION:
        model = LogisticRegression(max_iter=1000)
    elif model_type == DECISION_TREE:
        model = DecisionTreeClassifier()
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """
    Predict targets
    :param model: pretrained model
    :param features: test features like in train data
    :return: array of predictions
    """

    predicts = model.predict(features)
    return predicts


def evaluate_model(true_values: np.ndarray, predict_values: pd.Series) -> Dict[str, float]:
    """
    Evaluate model
    :param true_values: real classes
    :param predict_values: model predictions
    :return: dict with binary classification metrics
    """

    return {
        "accuracy": accuracy_score(true_values, predict_values),
        "precision": precision_score(true_values, predict_values),
        "recall": recall_score(true_values, predict_values)
    }


def create_inference_pipeline(model: classificationModels) -> Pipeline:
    """
    Create pipeline for easy use different models on next steps
    :param model: simple classification model
    :return: pipline consists of one step - model
    """

    return Pipeline([("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    """
    Dump model to file
    :param model: trained model
    :param output: output path to saved model
    :return: output path to saved model after successful saving
    """

    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
