import requests
import json


class MyHttpClient:

    def __init__(self, url):
        self.url = url

    def send_test_result(self, model_name, ppl):
        data = {
            "model_name": f"{model_name}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/test_result', json=data)

    def send_valid_result(self, model_name, ppl):
        data = {
            "model_name": f"{model_name}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_result', json=data)

    def send_model_size(self,  model_name, size):
        data = {
            "model_name": f"{model_name}",
            "size": size
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/model_size', json=data)

    def send_model_params(self,  model_name, params):
        data = {
            "model_name": f"{model_name}",
            "params": params
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/model_params', json=data)


