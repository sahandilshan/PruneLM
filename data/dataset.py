from train.corpus import Corpus
from train.utils import batchify


def get_dataset(device, batch_size):
    model_data_filepath = 'data/wikitext-2'
    corpus = Corpus(model_data_filepath)
    train_data = batchify(corpus.train, batch_size, device)
    valid_data = batchify(corpus.valid, batch_size, device)
    test_data = batchify(corpus.test, batch_size, device)
    return train_data, valid_data, test_data
