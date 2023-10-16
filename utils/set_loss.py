import torch


def choose_criterion(criterion_name):
    if criterion_name == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    elif criterion_name == "mse":
        return torch.nn.MSELoss()
    elif criterion_name == "bce":
        return torch.nn.BCELoss()
    else:
        raise ValueError(
            f"Loss function {criterion_name} not recognized. Please choose either 'crossentropy', 'mse', or 'bce'."
        )
