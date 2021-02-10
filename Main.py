import torch
from torch import optim, nn
import time
import math
from train.Corpus import Corpus
from train.Utils import batchify, evaluate
from model.Bi_LSTM_LM import Bi_LSTM_Model
from train.Train import train

# Set gpu if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_data_filepath = 'data/wikitext-2'
corpus = Corpus(model_data_filepath)

print(f'Number of words: {len(corpus.dictionary)}')
BATCH_SIZE = 256
SEQUENCE_LENGTH = 6
TRAIN_DATA = batchify(corpus.train, BATCH_SIZE, device)
VAL_DATA = batchify(corpus.valid, BATCH_SIZE, device)
TEST_DATA = batchify(corpus.test, BATCH_SIZE, device)
NUM_TOKENS = len(corpus.dictionary)
EMBEDDING_DIMS = 512
HIDDEN_DIMS = 256
NUM_LAYERS = 2
DROPOUT = 0.5
EPOCHS = 10
MODEL_SAVED_PATH = 'model.ckpt'

model = Bi_LSTM_Model(vocab_size=NUM_TOKENS, embedding_dims=EMBEDDING_DIMS,
                      hidden_dims=HIDDEN_DIMS, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
best_val_loss = None
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, criterion, optimizer, corpus, TRAIN_DATA, epoch, EPOCHS,
          batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)
    val_loss = evaluate(VAL_DATA, model, criterion, NUM_TOKENS, BATCH_SIZE, SEQUENCE_LENGTH)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        print('model saving....')
        torch.save(model.state_dict(), MODEL_SAVED_PATH)
        best_val_loss = val_loss
    else:
        print('Model not saving....')
