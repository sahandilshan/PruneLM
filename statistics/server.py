# import main Flask class and request object
from flask import Flask, request

# create the Flask app
app = Flask('Pruning Statistics')


@app.route('/test', methods=['POST', 'GET'])
def query_example():
    request_data = request.get_json()
    if request_data is not None:
        language = request_data['test']
        print(language)
        # framework = request_data['framework']
    return 'OK'


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=False, host='0.0.0.0', port=5000)
