import os

import pandas as pd
from fastapi import FastAPI
from joblib import load
from predict_input import PredictInput

app = FastAPI()

model = None


@app.on_event("startup")
def load_model():
    path_to_model = os.getenv('PATH_TO_MODEL')
    global model
    model = load(path_to_model)


@app.post("/predict")
async def predict(predict_input: PredictInput):
    data = pd.DataFrame([predict_input.dict()])
    prediction = model.predict(data)
    return prediction


@app.get('/health')
async def check_health():
    if model:
        return 200
