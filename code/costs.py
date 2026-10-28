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

def calculate_cross_entropy(y, tx, w):
    """Calculate the cross entropy"""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])

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
        return calculate_cross_entropy(y, tx, w)
    else :
        raise ValueError("Loss function should be mse or mae")

