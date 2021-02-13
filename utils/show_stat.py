def show_parameters_stats(total_params, pruned_model_params):
    print('Total Number of Parameters before Pruning:', total_params)
    print('After Pruning: ', pruned_model_params)
    dropped_params_count = total_params - pruned_model_params
    print('Dropped Parameters:', dropped_params_count)
    print(f'Pruned percentage'
          f': {str(round(dropped_params_count / total_params * 100, 2))}%')