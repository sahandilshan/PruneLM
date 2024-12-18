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
from utils.show_stat import show_parameters_stats, show_model_size_stats
from utils.size import get_original_model_size, get_pruned_model_size
from statistics.statistics_sender import MyHttpClient
from utils.validate_configs import validate_configs

config = configparser.RawConfigParser()
config.read('./configs/PruneLM.cfg')
validate_configs(config)
prune_configs = dict(config.items('Prune Configs'))
model_configs = dict(config.items('Model Configs'))
stat_configs = dict(config.items('Statistics Configs'))
print('Prune Configs:', prune_configs)
print('Model Load Configs:', model_configs)
print('Statistics Configs', stat_configs)

# Save Pruning Configs
PRUNING_ENABLED = prune_configs['enable']
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

# getting Statistics Configs
STAT_ENABLED = stat_configs['enable']
STAT_ENABLED = True if STAT_ENABLED == 'true' else False
SERVER_URL = stat_configs['server_url']
# PROMETHEUS_PORT = stat_configs['prometheus_port']
client = None
if STAT_ENABLED:
    client = MyHttpClient(SERVER_URL)

# Loading the model
DEVICE = model_configs['device']
NUM_TOKENS = int(model_configs['tokens'])
BATCH_SIZE = int(model_configs['batch_size'])
SEQUENCE_LENGTH = int(model_configs['sequence_length'])
EMBEDDING_DIMS = int(model_configs['embedding_dims'])
HIDDEN_DIMS = int(model_configs['hidden_dims'])
NUM_LAYERS = int(model_configs['num_stacked_rnn'])
DROPOUT = float(model_configs['dropout'])
PATH_TO_STATE_DIC = model_configs['model_path']
MODEL_SAVING_TYPE = model_configs.get('model_saving_type', 'best')
model = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                      hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.load_model(PATH_TO_STATE_DIC)
if PRUNING_ENABLED != 'true':
    print('skipping pruning and showing status of existing pruned models')
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
original_model_size = get_original_model_size(model)

if STAT_ENABLED:
    client.send_test_ppl('original', PRUNING_TYPE, math.exp(test_loss))
    client.send_valid_ppl('original', PRUNING_TYPE, math.exp(val_loss))
    # to get show the value in overall statistics dashboard
    client.send_final_valid_ppl('original', PRUNING_TYPE, math.exp(val_loss))
    client.send_model_size('original', PRUNING_TYPE, original_model_size)
    client.send_model_params('original', PRUNING_TYPE, total_params)

# Basic Pruning
if PRUNING_TYPE == 'basic' and PRUNING_ENABLED == 'true':
    for percentage in PERCENTAGES:
        prunedModel = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                    hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
        prunedModel.to(DEVICE)
        prune = Prune(model, percentage)
        pruned_state_dic = prune.modelPruning()
        prunedModel.load_state_dict(pruned_state_dic)
        pruned_model_params = get_pruned_parameters_count(prunedModel)
        show_parameters_stats(total_params, pruned_model_params)
        model_name = 'pruned_model_' + str(percentage) + '.ckpt'
        path = MODEL_SAVING_PATH + '/' + model_name
        torch.save(prunedModel.state_dict(), path)
        print('model saved.')
        print(f'Model saved path: {path}')
        # pruned_model_size = get_pruned_model_size(prunedModel)
        val_loss = evaluate(VALID_SET, prunedModel, criterion, NUM_TOKENS,
                            BATCH_SIZE, SEQUENCE_LENGTH)
        test_loss = evaluate(TEST_SET, prunedModel, criterion, NUM_TOKENS,
                             BATCH_SIZE, SEQUENCE_LENGTH)
        # print('-' * 89)
        print('| valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        print('| test loss {:5.2f} | '
              'test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        print('-' * 89)
        if STAT_ENABLED:
            pruned_model_size = get_pruned_model_size(prunedModel)
            client.init_pruned_model(model_name, PRUNING_TYPE)
            client.send_valid_ppl(model_name, PRUNING_TYPE, math.exp(val_loss))
            client.send_test_ppl(model_name, PRUNING_TYPE, math.exp(test_loss))
            client.send_model_size(model_name, PRUNING_TYPE, pruned_model_size)
            client.send_model_params(model_name, PRUNING_TYPE, pruned_model_params)

# Iterative Pruning
elif PRUNING_TYPE == 'iterative' and PRUNING_ENABLED == 'true':
    for percentage in PERCENTAGES:
        print('-' * 37, 'Pruning model from ' + str(percentage) + '%', '-' * 38)
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
        model_name = 'pruned_model_' + str(percentage) + '.ckpt'
        path = MODEL_SAVING_PATH + '/' + model_name
        total_time = 0
        if STAT_ENABLED:
            client.init_pruned_model(model_name, PRUNING_TYPE)
            client.send_total_epoch_size(model_name, PRUNING_TYPE, EPOCHS)
        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            train_loss = train(prunedModel, prune_criterion, prune_optimizer, NUM_TOKENS, TRAIN_SET,
                               epoch, EPOCHS, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
            prune = Prune(prunedModel, percentage)
            pruned_state_dic = prune.modelPruning()
            prunedModel.load_state_dict(pruned_state_dic)
            val_loss = evaluate(VALID_SET, prunedModel, prune_criterion, NUM_TOKENS,
                                BATCH_SIZE, SEQUENCE_LENGTH)
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            total_time += elapsed_time
            print('-' * 70)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, elapsed_time,
                                             val_loss, math.exp(val_loss)))
            print('-' * 70)
            if MODEL_SAVING_TYPE == 'best':
                if not best_val_loss or val_loss < best_val_loss:
                    print('model saving....')
                    torch.save(prunedModel.state_dict(), path)
                    best_val_loss = val_loss
                else:
                    print('Model not saving....')

            if STAT_ENABLED:
                client.send_last_epoch_number(model_name, PRUNING_TYPE, epoch)
                client.send_valid_loss(model_name, PRUNING_TYPE, val_loss)
                client.send_valid_ppl(model_name, PRUNING_TYPE, math.exp(val_loss))
                client.send_train_loss(model_name, PRUNING_TYPE, train_loss)
                client.send_train_ppl(model_name, PRUNING_TYPE, math.exp(train_loss))
                client.send_last_epoch_elapsed_time(model_name, PRUNING_TYPE, elapsed_time)
                client.send_last_epoch_finished_time(model_name, PRUNING_TYPE, epoch_end_time)
                client.send_total_elapsed_time(model_name, PRUNING_TYPE, total_time)

        if MODEL_SAVING_TYPE == 'last':
            torch.save(prunedModel.state_dict(), path)

        pruned_model_params = get_pruned_parameters_count(prunedModel)
        show_parameters_stats(total_params, pruned_model_params)
        print('-' * 100 + '\n')

        if STAT_ENABLED:
            prunedModel.load_model(path)
            pruned_model_size = get_pruned_model_size(prunedModel)
            test_loss = evaluate(TEST_SET, prunedModel, prune_criterion, NUM_TOKENS,
                                 BATCH_SIZE, SEQUENCE_LENGTH)
            valid_loss = evaluate(VALID_SET, prunedModel, prune_criterion, NUM_TOKENS,
                                  BATCH_SIZE, SEQUENCE_LENGTH)  # get the best validation loss
            client.send_test_ppl(model_name, PRUNING_TYPE, math.exp(test_loss))
            client.send_final_valid_ppl(model_name, PRUNING_TYPE, math.exp(valid_loss))
            client.send_model_size(model_name, PRUNING_TYPE, pruned_model_size)
            client.send_model_params(model_name, PRUNING_TYPE, pruned_model_params)

# Compression stat
for file in os.listdir(MODEL_SAVING_PATH):
    if file.endswith(".ckpt"):
        path = os.path.join(MODEL_SAVING_PATH, file)
        pruned_model = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                                     hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS,
                                     dropout=DROPOUT)
        pruned_model.to(DEVICE)
        pruned_model.load_model(path)
        pruned_model_size = get_pruned_model_size(pruned_model)
        print('-' * 37, file, '-' * 38)
        pruned_model_params = get_pruned_parameters_count(pruned_model)
        show_parameters_stats(total_params, pruned_model_params)
        print()
        show_model_size_stats(original_model_size, pruned_model_size)
        if STAT_ENABLED:
            client.send_model_size(file, PRUNING_TYPE, pruned_model_size)
            client.send_model_params(file, PRUNING_TYPE, pruned_model_params)
            if PRUNING_ENABLED != 'true':
                client.init_pruned_model(file, PRUNING_TYPE)
                prune_criterion = get_criterion()
                test_loss = evaluate(TEST_SET, pruned_model, prune_criterion, NUM_TOKENS,
                                     BATCH_SIZE, SEQUENCE_LENGTH)
                valid_loss = evaluate(VALID_SET, pruned_model, prune_criterion, NUM_TOKENS,
                                      BATCH_SIZE, SEQUENCE_LENGTH)
                client.send_test_ppl(file, PRUNING_TYPE, math.exp(test_loss))
                if PRUNING_TYPE == 'iterative':
                    client.send_final_valid_ppl(file, PRUNING_TYPE, math.exp(valid_loss))
                else:
                    client.send_valid_ppl(file, PRUNING_TYPE, math.exp(valid_loss))

                print('\n| valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(valid_loss, math.exp(valid_loss)))
                print('| test loss {:5.2f} | '
                      'test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
        print('-' * 100 + '\n')

print('Complete Execution')
