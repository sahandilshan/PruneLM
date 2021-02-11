def get_total_parameters_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dropped_parameters_count(pruned_model):
    dropped = 0
    for param in pruned_model.parameters():
        if param is not None:
            dropped += param.numel() - param.nonzero().size(0)
    return dropped
