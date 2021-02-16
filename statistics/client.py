import requests
import json


class MyHttpClient:

    def __init__(self, url):
        self.url = url

    def init_pruned_model(self, model_name, pruning_type):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}"
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/init', json=data)

    def send_test_ppl(self, model_name, pruning_type, ppl):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/test_ppl', json=data)

    def send_valid_ppl(self, model_name, pruning_type, ppl):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_ppl', json=data)

    def send_final_valid_ppl(self, model_name, pruning_type, ppl):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_ppl_final', json=data)

    def send_valid_loss(self, model_name, pruning_type, loss):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "loss": loss
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/valid_loss', json=data)

    def send_train_ppl(self, model_name, pruning_type, ppl):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "ppl": ppl
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/train_ppl', json=data)

    def send_train_loss(self, model_name, pruning_type, loss):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "loss": loss
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/train_loss', json=data)

    def send_total_epoch_size(self, model_name, pruning_type, epochs):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "epochs": epochs
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/total_epochs', json=data)

    def send_current_epoch_number(self, model_name, pruning_type, epoch):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "epoch": epoch
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/current_epoch', json=data)

    def send_last_epoch_finished_time(self, model_name, pruning_type, time):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "time": time
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/last_epoch_finished_time', json=data)

    def send_last_epoch_elapsed_time(self, model_name, pruning_type, time):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "time": time
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/last_epoch_elapsed_time', json=data)

    # def send_total_batch_size(self, model_name, pruning_type, batch_size):
    #     data = {
    #         "model_name": f"{model_name}",
    #         "pruning_type": f"{pruning_type}",
    #         "batch_size": batch_size
    #     }
    #     data = json.dumps(data)
    #     requests.post(url=self.url + '/total_batch_size', json=data)
    #
    # def send_current_batch_number(self, model_name, pruning_type, batch_size):
    #     data = {
    #         "model_name": f"{model_name}",
    #         "pruning_type": f"{pruning_type}",
    #         "batch_size": batch_size
    #     }
    #     data = json.dumps(data)
    #     requests.post(url=self.url + '/current_batch', json=data)

    def send_model_size(self,  model_name, pruning_type, model_size):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "model_size": model_size
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/model_size', json=data)

    def send_model_params(self,  model_name, pruning_type, params):
        data = {
            "model_name": f"{model_name}",
            "pruning_type": f"{pruning_type}",
            "params": params
        }
        data = json.dumps(data)
        requests.post(url=self.url + '/model_params', json=data)


