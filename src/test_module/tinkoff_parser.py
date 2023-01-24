import requests
import yaml
from fake_useragent import UserAgent
SUCCESS_STATUS_CODE = 200

with open('src/test_module/config.yml') as c:
    config = yaml.safe_load(c)



class TinkoffTaker:
    def __init__(self, taker_type) -> None:
        headers = config[taker_type]["HEADERS"]
        headers["User-Agent"] = UserAgent().random
        params = config[taker_type]["PARAMS"]
        prefix_url = config[taker_type]["PREFIX_URL"]
        postfix_url = config[taker_type]["POSTFIX_URL"]

        self._session = requests.Session()
        self._session.headers.update(headers)
        self._session.params.update(params)
        self._prefix_url = prefix_url
        self._postfix_url = postfix_url

    def get_data(self, ticker, **kwargs):
        url = self._prefix_url + ticker + self._postfix_url
        response = self._session.get(url=url, params=kwargs)
        
        if response.status_code != SUCCESS_STATUS_CODE:
            return {}
        return response.json()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()



