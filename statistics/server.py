# import main Flask class and request object
from flask import Flask, request
from prometheus_client import start_http_server, Gauge

# create the Flask app
app = Flask('Pruning Statistics')


class Metrics(object):
    def __init__(self):
        self.test_result_gauge = Gauge('prune_lm_test_result',
                                       'show test results', ['model_name', 'model_type'])
        self.validation_result_gauge = Gauge('prune_lm_valid_result',
                                             'show test results', ['model_name', 'model_type'])


metrics = Metrics()


def start_prometheus_server(port):
    start_http_server(port)


@app.route('/', methods=['POST', 'GET'])
def init():
    start_http_server(8000)
    return 'OK'


@app.route('/test_result', methods=['POST'])
def published_test_result_metric():
    request_data = request.get_json()
    if request_data is not None:
        model_name = request_data['model_name']
        ppl = request_data['ppl']
        metrics.test_result_gauge.labels(model_name=model_name).set(ppl)
    return 'OK'


@app.route('/valid_result', methods=['POST'])
def published_test_result_metric():
    request_data = request.get_json()
    if request_data is not None:
        model_name = request_data['model_name']
        ppl = request_data['ppl']
        metrics.validation_result_gauge.labels(model_name).set(ppl)
    return 'OK'




if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, host='0.0.0.0', port=5000)
