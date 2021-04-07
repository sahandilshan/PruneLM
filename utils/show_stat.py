def show_parameters_stats(total_params, pruned_model_params):
    print('Total Number of Parameters before Pruning:', total_params)
    print('After Pruning: ', pruned_model_params)
    dropped_params_count = total_params - pruned_model_params
    print('Dropped Parameters:', dropped_params_count)
    print(f'Pruned percentage'
          f': {round(dropped_params_count / total_params * 100)}%')


def show_model_size_stats(original_model_size, pruned_model_size):
    print(f'Original Model size: {original_model_size}MB, '
          f'Pruned Model size: {pruned_model_size}MB')
    compressed_size = round(original_model_size - pruned_model_size, 2)
    print(f'Reduced size:{compressed_size}MB')
    print(f'Compressed Percentage'
          f': {round(compressed_size / original_model_size * 100, 2)}%')
