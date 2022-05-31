import torch


def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    """
    Mean Squared error with L2 regularization
    Args:
        inputs: Input values
        targets: Output targets
        model: Model object
        lamda: L2 lamda hyper-parameter

    Returns: regularized mse loss

    """
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2) / 2
    reg_loss = lamda * reg_loss
    mse_loss = torch.sum((inputs - targets) ** 2) / 2
    return mse_loss + reg_loss
