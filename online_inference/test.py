import json

import pytest as pytest
from fastapi.testclient import TestClient

from main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def start_test():
    load_model()


def test_predict():
    request = {"age": 48,
               "sex": 0,
               "cp": 2,
               "trestbps": 20,
               "chol": 100,
               "fbs": 0,
               "restecg": 2,
               "thalach": 150,
               "exang": 0,
               "oldpeak": 8,
               "slope": 1,
               "ca": 1,
               "thal": 1
               }

    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "no disease"}


def test_health_model():
    response = client.get('/health')
    assert response.status_code == 200


def wrong_input():
    request = {"age": 48,
               "sex": 2,
               "cp": 2,
               "trestbps": 20,
               "chol": 100,
               "fbs": 0,
               "restecg": 2,
               "thalach": 150,
               "exang": 0,
               "oldpeak": 8,
               "slope": 1,
               "ca": 1,
               "thal": 1
               }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == "age should be in 1-100"


def wrong_float_input():
    request = {"age": 101,
               "sex": 2,
               "cp": 2,
               "trestbps": 20,
               "chol": 100,
               "fbs": 0,
               "restecg": 2,
               "thalach": 150,
               "exang": 0,
               "oldpeak": 8,
               "slope": 1,
               "ca": 1,
               "thal": 1
               }
    response = client.post(
        url='/predict',
        content=json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == "unexpected value; permitted: 0, 1"
