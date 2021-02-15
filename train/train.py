from torch import nn
from tqdm import tqdm
import math
# import time
from train.utils import get_batch, repackage_hidden


def train(model, criterion, optimizer, num_tokens, train_data, epoch_no, epochs,
          batch_size=256, sequence_length=6, client=None, model_name=None):
    # Turn on training mode which enables dropout.
    assert num_tokens is not None
    model.train()
    total_loss = 0.
    loop = tqdm(enumerate(range(0, train_data.size(0) - 1, sequence_length)),
                total=len(train_data) // sequence_length, position=0, leave=True)
    counter = 0
    for batch, i in loop:
        data, targets = get_batch(train_data, i)
        hidden = model.init_hidden(batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        # print('data:', data.shape)
        # print('target:', targets.shape)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, num_tokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        total_loss += loss.item()
        counter += 1
        loop.set_description(f"Epoch: [{epoch_no}/{epochs}]")
        loop.set_postfix(loss=loss.item(), ppl=math.exp(loss.item()))

        if client is not None:
            client.send_current_batch_number(model_name, data)
            client.send_total_batch_size(model_name, len(train_data) // sequence_length)
            # avg_train_loss = total_loss / len(train_data)
            client.send_train_loss(model_name, loss.item())
            client.send_train_ppl(model_name, math.exp(math.exp(loss.item())))

