import torch


def batchify(data, batch_size, device='cpu'):
    # Get number of batches
    num_batch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, num_batch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, sequence_length=6):
    seq_len = min(sequence_length, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source, model, criterion, num_tokens, batch_size, sequence_length=6):
    model.eval()  # Stop calculating gradients
    total_loss = 0.
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, sequence_length):
            data, targets = get_batch(data_source, i, sequence_length)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, num_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)
