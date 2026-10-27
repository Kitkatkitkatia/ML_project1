import numpy as np

def sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_cross_entropy(y, y_prob):
    return - np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))

def compute_loss(y, tx, w, lf="mse"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        lf: string. Loss function, either mse or mae. 

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    if lf=="mse":
        return calculate_mse(e)
    elif lf=="mae":
        return calculate_mae(e)
    elif lf == "cross_entropy":
        y_prob = sigmoid(tx.dot(w))
        return calculate_cross_entropy(y, y_prob)
    else :
        raise ValueError("Loss function should be mse or mae")

