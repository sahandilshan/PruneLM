import torch
import torch.nn as nn

EMBEDDING_DIMS = 512
HIDDEN_DIMS = 256
NUM_LAYERS = 2
DROPOUT = 0.5

# Set gpu if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Bi_LSTM_Model:
    def __init__(self, vocab_size, embedding_dims, hidden_dims, num_layers, dropout=0.5):
        super(Bi_LSTM_Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.bi_lstm = nn.LSTM(embedding_dims, hidden_dims, num_layers,
                               bidirectional=True, dropout=dropout)
        # Since this is Bi-Directional, returns hidden_dims * 2 times parameters
        self.decoder = nn.Linear(hidden_dims * 2, vocab_size)
        self.init_weights()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.embedding(input))
        self.bi_lstm.flatten_parameters()
        output, hidden = self.bi_lstm(emb, hidden)
        output = self.drop(output)

        # shape: {(seq * batch_size), hidden_steps} - hidden state contains
        # num_directions * hidden_size
        output_ = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(output_)
        # Converting in to (seq, batch_size, vocab_size) dim vector
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())  # Calling super class method
        return (weight.new_zeros(self.num_layers * 2, bsz, self.hidden_dims),
                weight.new_zeros(self.num_layers * 2, bsz, self.hidden_dims))
