import requests
import json


class MyHttpClient:

    def __init__(self, url):
        self.url = url

    def send_test_result(self, model_type, ppl):
        data = {
            "model_type": f"{model_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/test_result', json=data)

    def send_valid_result(self, model_type, ppl):
        data = {
            "model_type": f"{model_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_result', json=data)


