import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, lf="mse"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        ls: string. Loss function, either mse or mae. 

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    if lf=="mse":
        return calculate_mse(e)
    elif lf=="mae":
        return calculate_mae(e)
    else :
        raise ValueError("Loss function should be mse or mae")