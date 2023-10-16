import torch


def choose_optimizer(optimizer_name, model, learning_rate, weight_decay, momentum):
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            # weight_decay=weight_decay,
            # momentum=momentum,
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(
            f"Optimizer {optimizer_name} not recognized. Please choose either 'sgd', 'adam', or 'rmsprop'."
        )
