import configparser
import math
import os
import time
import torch

from data.dataset import get_dataset
from model.bi_lstm_lm import Bi_LSTM_Model, get_criterion, get_optimizer
from prune.prune import Prune
from train.train import train
from train.utils import evaluate
from utils.parameters import get_total_parameters_count, get_pruned_parameters_count
from utils.size import get_original_model_size, get_pruned_model_size

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
    EPOCHS = int(prune_configs.get('epochs', 10))
    print(f'Epochs: {EPOCHS}')
print(f'Prune Type: {PRUNING_TYPE}, Percentage(s): {PERCENTAGES}')

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
MODEL_SAVING_TYPE = model_load_configs.get('model_saving_type', 'best')
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
print('-' * 36, 'original model performance', '-' * 36)
print('| valid loss {:5.2f} | '
      'valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
print('| test loss {:5.2f} | '
      'test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('-' * 100)

total_params = get_total_parameters_count(model)
# Basic Pruning
if PRUNING_TYPE == 'basic':
    for percentage in PERCENTAGES:
        prunedModel = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                    hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
        prunedModel.to(DEVICE)
        prune = Prune(model, percentage)
        pruned_state_dic = prune.modelPruning()
        prunedModel.load_state_dict(pruned_state_dic)
        pruned_model_params = get_pruned_parameters_count(prunedModel)
        print('Total Number of Parameters before Pruning:', total_params)
        print('Dropped Parameters:', (total_params - pruned_model_params))
        print('After Pruning: ', pruned_model_params)
        print(f'Percentage: {str(round(pruned_model_params / total_params * 100, 2))}%')
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

# Iterative Pruning
elif PRUNING_TYPE == 'iterative':
    for percentage in PERCENTAGES:
        print('-' * 37, 'Pruning model from' + str(percentage) + '%', '-' * 38)
        best_val_loss = None
        prunedModel = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                    hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
        prunedModel.to(DEVICE)
        prune_optimizer = get_optimizer(prunedModel)
        prune_criterion = get_criterion()
        # Pruning for the first time before begins training
        prune = Prune(model, percentage)
        pruned_state_dic = prune.modelPruning()
        prunedModel.load_state_dict(pruned_state_dic)
        path = MODEL_SAVING_PATH + '/pruned_model_' + str(percentage) + '.ckpt'
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train(prunedModel, prune_criterion, prune_optimizer, NUM_TOKENS, TRAIN_SET, epoch, EPOCHS,
                  batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
            prune = Prune(prunedModel, percentage)
            pruned_state_dic = prune.modelPruning()
            prunedModel.load_state_dict(pruned_state_dic)
            val_loss = evaluate(VALID_SET, prunedModel, prune_criterion, NUM_TOKENS,
                                BATCH_SIZE, SEQUENCE_LENGTH)
            print('-' * 70)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 70)
            if MODEL_SAVING_TYPE == 'best':
                if not best_val_loss or val_loss < best_val_loss:
                    print('model saving....')
                    torch.save(prunedModel.state_dict(), path)
                    best_val_loss = val_loss
                else:
                    print('Model not saving....')

        if MODEL_SAVING_TYPE == 'last':
            torch.save(prunedModel.state_dict(), path)

        pruned_model_params = get_pruned_parameters_count(prunedModel)
        print('Total Number of Parameters before Pruning:', total_params)
        print('After Pruning: ', pruned_model_params)
        dropped_params_count = total_params - pruned_model_params
        print('Dropped Parameters:', dropped_params_count)
        print(f'Pruned percentage'
              f': {str(round(dropped_params_count / total_params * 100, 2))}%')
        print('-' * 100 + '\n')

# Compression stat
for file in os.listdir(MODEL_SAVING_PATH):
    original_model_size = get_original_model_size(model)
    if file.endswith(".ckpt"):
        path = os.path.join(MODEL_SAVING_PATH, file)
        pruned_model = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                     hidden_dims=HIDDEN_DIMS, num_layers=2, dropout=DROPOUT)
        pruned_model.load_model(path)
        pruned_model_size = get_pruned_model_size(pruned_model)
        print('-' * 37, file + '%', '-' * 38)
        print(f'Original Model size: {original_model_size}, '
              f'Pruned Model size: {pruned_model_size}')
        compressed_size = round(original_model_size - pruned_model_size, 2)
        print(f'Reduced size:{compressed_size}')
        print(f'Compressed Percentage'
              f': {round(compressed_size / original_model_size * 100, 2)}')
        print('-' * 100 + '\n')

print('done')
