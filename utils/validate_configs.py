# Check if configs has only Prune, Model, Statistics and Prometheus configs
def validate_configs(configs):
    supported_configs = ['Prune Configs', 'Model Configs',
                         'Statistics Configs', 'Prometheus Configs']
    configs_keys = list(dict(configs.items()).keys())
    configs_keys.pop(0)  # to remove the "DEFAULT key
    for config in configs_keys:
        if config not in supported_configs:
            raise ImportError(f'{config}: is not a supported configuration. '
                              f'Supported Configurations: {supported_configs}')
