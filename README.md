# PruneLM
Compressed and Optimized Language Models with Bi-LSTM Architectures

# How to Setup
1. First clone this repository.
2. Install required pip packages by issuing following command
   `pip install -r requirements.txt`
3. Navigate to `config` directory and open the `PruneLM.cfg` file.
4. Provide the necessary information needed for the compression. (The details about how to provide relevant configs are listed in the `PruneLM.cfg`)
5. Navigate back to the base directory.
6. Start the compression by running the `main.py` file.
   
# How to use the Statistics Dashboards
1. Download and install Prometheus from [this](https://prometheus.io/download/) URL. (**Note: Select the relevant OS type before downloading)
2. Download and install Grafana from [this](https://grafana.com/get/?plcmt=top-nav&cta=downloads) url. (**Note: Select the relevant OS type before downloading)
3. Run the prometheus server.
4. Start the Grafana service and navigate to Grafana Home with `localhost:3000` URL.
5. Then import the dashboards provided in the `statistics/dashboard` directory.