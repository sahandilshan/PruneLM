import requests
import json


class MyHttpClient:

    def __init__(self, url):
        self.url = url

    def init_pruned_model(self, model_name):
        data = {
            "model_name": f"{model_name}"
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/init', json=data)

    def send_test_ppl(self, model_name, ppl):
        data = {
            "model_name": f"{model_name}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/test_ppl', json=data)

    def send_valid_ppl(self, model_name, ppl):
        data = {
            "model_name": f"{model_name}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_ppl', json=data)

    def send_valid_loss(self, model_name, loss):
        data = {
            "model_name": f"{model_name}",
            "loss": loss
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_loss', json=data)

    def send_train_ppl(self, model_name, ppl):
        data = {
            "model_name": f"{model_name}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/train_ppl', json=data)

    def send_train_loss(self, model_name, loss):
        data = {
            "model_name": f"{model_name}",
            "loss": loss
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/train_loss', json=data)

    def send_total_epoch_size(self, model_name, epochs):
        data = {
            "model_name": f"{model_name}",
            "epochs": epochs
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/total_epochs', json=data)

    def send_current_epoch_number(self, model_name, epoch):
        data = {
            "model_name": f"{model_name}",
            "epoch": epoch
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/current_epoch', json=data)

    def send_last_epoch_finished_time(self, model_name, time):
        data = {
            "model_name": f"{model_name}",
            "time": time
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/last_epoch_finished_time', json=data)

    def send_last_epoch_elapsed_time(self, model_name, time):
        data = {
            "model_name": f"{model_name}",
            "time": time
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/last_epoch_elapsed_time', json=data)

    def send_total_batch_size(self, model_name, batch_size):
        data = {
            "model_name": f"{model_name}",
            "batch_size": batch_size
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/total_batch_size', json=data)

    def send_current_batch_number(self, model_name, batch_size):
        data = {
            "model_name": f"{model_name}",
            "batch_size": batch_size
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/current_batch', json=data)

    def send_model_size(self,  model_name, model_size):
        data = {
            "model_name": f"{model_name}",
            "model_size": model_size
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


