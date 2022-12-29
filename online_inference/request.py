import json
import logging

import pandas as pd
import requests

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)


def main():
    logger.info("Start prediction.")
    data = pd.read_csv('data.csv')
    data = data.drop('condition', axis=1)
    request_data = data.to_dict(orient='records')

    for request in request_data:
        response = requests.post(
            url='http://0.0.0.0:8888/predict',
            data=json.dumps(request))
        logger.info(f'Request stats code: {response.status_code}')
        logger.info(f'Response: {response}')
    logger.info("Finish prediction.")


if __name__ == "__main__":
    main()
