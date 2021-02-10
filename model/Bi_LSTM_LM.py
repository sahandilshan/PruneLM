import torch.nn as nn


class Bi_LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_dims, num_layers, dropout=0.5):
        super(Bi_LSTM_Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.bi_lstm = nn.LSTM(embedding_dims, hidden_dims, num_layers,
                               bidirectional=True, dropout=dropout)
        # Since this is Bi-Directional, returns hidden_dims * 2 times parameters
        self.output_layer = nn.Linear(hidden_dims * 2, vocab_size)
        self.init_weights()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.embedding(input))
        self.bi_lstm.flatten_parameters()
        output, hidden = self.bi_lstm(emb, hidden)
        output = self.drop(output)
        output_size_0 = output.size(0)     # sequence size
        output_size_1 = output.size(1)     # batch size
        output_size_2 = output.size(2)     # hidden_size

        # shape: {(seq * batch_size), hidden_dims} - hidden state contains
        # num_directions * hidden_size
        output = output.view(output_size_0 * output_size_1, output_size_2)
        output = self.output_layer(output)
        # Converting in to (seq, batch_size, vocab_size) dim vector
        output = output.view(output_size_0, output_size_1, output.size(1))
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())  # Calling super class method
        return (weight.new_zeros(self.num_layers * 2, bsz, self.hidden_dims),
                weight.new_zeros(self.num_layers * 2, bsz, self.hidden_dims))
