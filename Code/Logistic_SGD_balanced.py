from Code.implementations import grad_logistic, sigmoid
from Code.helpers import batch_iter
from Code.costs import compute_loss
from Code.evaluation import *
from Code.helpers import build_k_indices

import numpy as np

from Code.evaluation import *

def stochastic_logistic_regression_balanced(y, tx, initial_w, batch_size, max_iters,
                                       gamma, lambda_, balanced=True, seed=0):
    """cross validation over regularisation parameter lambda and training rate gamma.
    
        Args:
            y: array, target labels
            tx: array, features 
            inital_w: array, start weights
            batch_size: scalar, nb of sampled data for gradient computation
            max_iters: scalar, nb of iterations of SGD
            gamma: scalar, learning rate
            lambda: scalar, regularization strength
            balanced: bool, True force balanced class
        Returns:
            loss_tr : integer, value of the best gamma
            loss_te : scalar, value of the best lambda
            f1s : value of the cross entropy loss for the couple (best_gamma, best_lambda)
            ts : array, decisions threshold for logistic regression
        """
    rng = np.random.default_rng(seed)
    w = initial_w
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = batch_size // 2

    for _ in range(max_iters):
        if balanced:
            idx = np.concatenate([rng.choice(pos, half), rng.choice(neg, half)]) # force balanced
        else:
            idx = rng.choice(len(y), batch_size, replace=False)

        reg = lambda_ * w
        reg[0] = 0 # no regularization of bias term
        grad = grad_logistic(y[idx], tx[idx], w) + reg
        w = w - gamma * grad

    loss = compute_loss(y, tx, w, "cross_entropy")
    return w, loss

def cross_validation_logreg_balanced(y, x, k_indices, k, gamma, max_iters, batch_size, lambda_):
    """cross validation over regularisation parameter lambda and training rate gamma.
        
        Args:
            y: array, target labels
            tx: array, features 
            inital_w: array, start weights
            batch_size: scalar, nb of sampled data for gradient computation
            max_iters: scalar, nb of iterations of SGD
            gamma: scalar, learning rate
            lambda: scalar, regularization strength
            balanced: bool, True force balanced class
        Returns:
            loss_tr : integer, value of the best gamma
            loss_te : scalar, value of the best lambda
            f1s : value of the cross entropy loss for the couple (best_gamma, best_lambda)
            ts : array, decisions threshold for logistic regression
    """

    te = k_indices[k]
    tr = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)
    w, loss_tr = stochastic_logistic_regression_balanced(
        y[tr], x[tr], np.zeros(x[tr].shape[1]), batch_size, max_iters, gamma, lambda_
    )

    # validation fold stays still unbalanced
    # tuning threshold
    p = sigmoid(x[te] @ w)
    ts = np.linspace(0.05, 0.95, 46)
    f1s = [f1_score(y[te], (p >= t).astype(int)) for t in ts]
    i = int(np.argmax(f1s))
    loss_te = compute_loss(y[te], x[te], w, "cross_entropy")

    return float(loss_tr), float(loss_te), float(f1s[i]), float(ts[i])



def cross_validation_logreg_demo(y, x, k_fold, gammas, max_iters, batch_size, lambda_):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """


    print("Unique y values:", np.unique(y))
    print("x mean, std min/max:", np.mean(x), np.min(np.std(x, axis=0)), np.max(np.std(x, axis=0)))

    seed = 12
    k_fold = k_fold
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    ### SOLUTION
    # cross validation
    for gamma in gammas:
        print("Computing gamma=", gamma, "----------------------------")
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            print("Computing gamma=", gamma, 'fold: ',k, "#######################")
            loss_tr, loss_te = cross_validation_logreg(y, x, k_indices, k, gamma, max_iters, batch_size, lambda_)
            diff = abs(loss_tr - loss_te)
            print(f"Train: {loss_tr:.6f} | Test: {loss_te:.6f} | Absolute Diff: {diff:.6e}")
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    best_gamma, best_rmse = gammas[np.argmin(rmse_te)], np.min(rmse_te)
 

    #cross_validation_visualization(gammas, rmse_tr, rmse_te)
    print(
        "For SGD, the choice of gamma which leads to the best test mse is %.5f with a test mse of %.3f"
        % (best_gamma, best_rmse)
    )
    return best_gamma, best_rmse