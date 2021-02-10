from train.Corpus import Corpus
from train.Utils import batchify
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_data_filepath = 'data/wikitext-2'
corpus = Corpus(model_data_filepath)
print(f'Number of words: {len(corpus.dictionary)}')
batch_size = 256
train_data = batchify(corpus.train, batch_size, device)
val_data = batchify(corpus.valid, batch_size, device)
test_data = batchify(corpus.test, batch_size, device)

