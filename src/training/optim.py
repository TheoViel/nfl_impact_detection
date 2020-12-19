import torch


def define_optimizer(name, params, lr=1e-3):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.

    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.

    Raises:
        NotImplementedError: Specified optimizer name is not supported.

    Returns:
        torch optimizer: Optimizer
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr)
    except AttributeError:
        raise NotImplementedError

    return optimizer
