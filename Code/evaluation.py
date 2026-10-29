import numpy as np

def accuracy(y,pred):
    """Compute accurary of prediction"""
    return np.sum(y == pred)/pred.shape[0]

def f1_score(y, pred):
    """Compute F1 score of prediction"""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return 2 * tp / (2 * tp + fp + fn + 1e-12)