FROM python:3.10.8

COPY requirements.txt ./online_inference/
WORKDIR /online_inference
RUN python3 -m pip install --upgrade pip && pip install -r ./requirements.txt

COPY . .

ENV PATH_TO_MODEL=models/model.pkl

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]

