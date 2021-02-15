import configparser
import json
from flask import Flask, request
from prometheus_client import Gauge, Counter, start_http_server

# create the Flask app
app = Flask('Pruning Statistics')

config = configparser.RawConfigParser()
config.read('../configs/pruningConfigs.cfg')
stat_configs = dict(config.items('Prometheus Configs'))


def create_prometheus_client(port):
    start_http_server(int(port))


class Metrics(object):
    def __init__(self):
        port = stat_configs['prometheus_port']
        start_http_server(int(port))
        self.init_pruned_metric = Counter('prune_lm_pruned_model',
                                          'to get the names of pruned models',
                                          ['model_name', 'pruning_type'])
        self.test_ppl_gauge = Gauge('prune_lm_test_ppl',
                                    'show test perplexity', ['model_name', 'pruning_type'])
        self.validation_ppl_gauge = Gauge('prune_lm_valid_ppl',
                                          'show test perplexity', ['model_name', 'pruning_type'])
        self.valid_loss_gauge = Gauge('prune_lm_valid_loss',
                                      'validation loss of current epoch',
                                      ['model_name', 'pruning_type'])
        self.train_loss_gauge = Gauge('prune_lm_train_loss',
                                      'training loss of the current epoch',
                                      ['model_name', 'pruning_type'])
        self.train_ppl_gauge = Gauge('prune_lm_train_ppl', 'show train_perplexity',
                                     ['model_name', 'pruning_type'])
        self.last_epoch_finished_gauge = Gauge('prune_lm_last_epoch_finished',
                                               'finished time of the last_epoch',
                                               ['model_name', 'pruning_type'])
        self.last_epoch_elapsed_time_gauge = Gauge('prune_lm_last_epoch_elapsed',
                                                   'elapsed time of the last epoch',
                                                   ['model_name', 'pruning_type'])
        self.current_epoch_counter = Counter('prune_lm_current_epoch', 'current epoch',
                                             ['model_name', 'pruning_type'])
        self.total_epoch_gauge = Gauge('prune_lm_total_epoch', 'total epoch size',
                                       ['model_name', 'pruning_type'])
        self.total_batch_size_gauge = Gauge('prune_lm_total_batch_size',
                                            'total batch size of an epoch',
                                            ['model_name', 'pruning_type'])
        self.current_batch_gauge = Gauge('prune_lm_current_batch',
                                         'current batch of the epoch', ['model_name', 'pruning_type'])


metrics = Metrics()


@app.route('/', methods=['POST', 'GET'])
def init():
    return 'OK'


@app.route('/init', methods=['POST'])
def published_init_metrics():  # this metric is only used to get the pruned_model name as variable
    msg_body = request.get_json()
    msg_body = json.loads(msg_body)
    if msg_body is not None:
        model_name = msg_body['model_name']
        pruning_type = msg_body['pruning_type']
        metrics.init_pruned_metric.labels(model_name, pruning_type).inc()
    return 'OK'


@app.route('/test_ppl', methods=['POST'])
def published_test_ppl_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        ppl = request_data['ppl']
        metrics.test_ppl_gauge.labels(model_name, pruning_type).set(ppl)
    return 'OK'


@app.route('/valid_ppl', methods=['POST'])
def published_valid_ppl_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        ppl = request_data['ppl']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(ppl)
    return 'OK'


@app.route('/valid_loss', methods=['POST'])
def published_valid_loss_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        loss = request_data['loss']
        metrics.valid_loss_gauge.labels(model_name, pruning_type).set(loss)
    return 'OK'


@app.route('/train_ppl', methods=['POST'])
def published_train_ppl_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        ppl = request_data['ppl']
        metrics.train_ppl_gauge.labels(model_name, pruning_type).set(ppl)
    return 'OK'


@app.route('/train_loss', methods=['POST'])
def published_test_loss_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        loss = request_data['loss']
        metrics.train_loss_gauge.labels(model_name, pruning_type).set(loss)
    return 'OK'


@app.route('/total_epochs', methods=['POST'])
def published_total_epochs_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        epochs = request_data['epochs']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(epochs)
    return 'OK'


@app.route('/current_epoch', methods=['POST'])
def published_current_epoch_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        epoch = request_data['epoch']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(epoch)
    return 'OK'


@app.route('/last_epoch_finished_time', methods=['POST'])
def published_last_epoch_finished_time_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        time = request_data['time']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(time)
    return 'OK'


@app.route('/last_epoch_elapsed_time', methods=['POST'])
def published_last_epoch_elapsed_time_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        time = request_data['time']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(time)
    return 'OK'


@app.route('/total_batch_size', methods=['POST'])
def published_total_batch_size_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        batch_size = request_data['batch_size']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(batch_size)
    return 'OK'


@app.route('/current_batch', methods=['POST'])
def published_current_batch_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        batch_size = request_data['batch_size']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(batch_size)
    return 'OK'


@app.route('/model_size', methods=['POST'])
def published_model_size_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        model_size = request_data['model_size']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(model_size)
    return 'OK'


@app.route('/model_params', methods=['POST'])
def published_model_params_metric():
    request_data = request.get_json()
    request_data = json.loads(request_data)
    if request_data is not None:
        model_name = request_data['model_name']
        pruning_type = request_data['pruning_type']
        params = request_data['params']
        metrics.validation_ppl_gauge.labels(model_name, pruning_type).set(params)
    return 'OK'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, host='0.0.0.0', port=5000)
