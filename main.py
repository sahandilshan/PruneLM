import configparser
import os
import math
from model.bi_lstm_lm import Bi_LSTM_Model, get_criterion, get_optimizer
from prune.prune import Prune
from prune.utils import *
from data.dataset import get_dataset
from train.utils import evaluate

config = configparser.RawConfigParser()
config.read('configs/pruningConfigs.cfg')
prune_configs = dict(config.items('Prune Configs'))
model_load_configs = dict(config.items('Model Loading Configs'))
print(prune_configs)
print(model_load_configs)

# Save Pruning Configs
PRUNING_TYPE = prune_configs['prune_type']
PERCENTAGES = prune_configs['percentage']
PERCENTAGES = PERCENTAGES.split()
PERCENTAGES = list(map(int, PERCENTAGES))
MODEL_SAVING_PATH = prune_configs['model_saving_directory']
try:
    os.makedirs(MODEL_SAVING_PATH)
except FileExistsError:
    # directory already exists
    pass
EPOCHS = None
if PRUNING_TYPE == 'iterative':
    EPOCHS = prune_configs['epochs']
    print(f'Epochs: {EPOCHS}')
print(f'Prune Type: {PRUNING_TYPE}, Percentage: {PERCENTAGES}')

# Loading the model
DEVICE = model_load_configs['device']
NUM_TOKENS = int(model_load_configs['tokens'])
BATCH_SIZE = int(model_load_configs['batch_size'])
SEQUENCE_LENGTH = int(model_load_configs['sequence_length'])
EMBEDDING_DIMS = int(model_load_configs['embedding_dims'])
HIDDEN_DIMS = int(model_load_configs['hidden_dims'])
NUM_LAYERS = int(model_load_configs['num_stacked_rnn'])
DROPOUT = float(model_load_configs['dropout'])
PATH_TO_STATE_DIC = model_load_configs['model_path']
model = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                      hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.load_model(PATH_TO_STATE_DIC)
model.to(DEVICE)
optimizer = get_optimizer(model)
criterion = get_criterion()

# Loading the dataset (train, validation, test)
TRAIN_SET, VALID_SET, TEST_SET = get_dataset(DEVICE, BATCH_SIZE)

# Evaluate the model
val_loss = evaluate(VALID_SET, model, criterion, NUM_TOKENS,
                    BATCH_SIZE, SEQUENCE_LENGTH)
test_loss = evaluate(TEST_SET, model, criterion, NUM_TOKENS,
                     BATCH_SIZE, SEQUENCE_LENGTH)
print('-' * 89)
print('| valid loss {:5.2f} | '
      'valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
print('| test loss {:5.2f} | '
      'test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('-' * 89)

# Basic Pruning
if PRUNING_TYPE == 'basic':
    total_params = get_total_parameters_count(model)
    for percentage in PERCENTAGES:
        prunedModel = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                    hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
        prunedModel.to(DEVICE)
        prune = Prune(model, percentage)
        pruned_state_dic = prune.modelPruning()
        prunedModel.load_state_dict(pruned_state_dic)
        dropped_params_count = get_dropped_parameters_count(prunedModel)
        print('Total Number of Parameters before Pruning:', total_params)
        print('Dropped Parameters:', dropped_params_count)
        print('After Pruning: ', (total_params - dropped_params_count))
        print(f'Percentage: {(dropped_params_count / total_params) * 100}%')
        path = MODEL_SAVING_PATH + '/pruned_model_' + str(percentage) + '.ckpt'
        torch.save(prunedModel.state_dict(), path)
        print('model saved.')
        print(f'Model saved path: {path}')
        val_loss = evaluate(VALID_SET, prunedModel, criterion, NUM_TOKENS,
                            BATCH_SIZE, SEQUENCE_LENGTH)
        print('-' * 89)
        print('| valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        print('-' * 89)

print('done')
