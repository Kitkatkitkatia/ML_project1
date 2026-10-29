from Code.implementations import grad_logistic
from Code.helpers import batch_iter
from Code.costs import compute_loss
from plots import cross_validation_visualization
from Code.helpers import build_k_indices

import numpy as np

def stochastic_logistic_regression_reg(y, tx, initial_w, batch_size, max_iters, gamma, lambda_):
    """Logistic regression using stochastic gradient descent (SGD).

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape(D,), D is the number of features. The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        lambda_ : a scalar denoting the impact of L2 regularization

    Returns:
        loss: loss value (scalar) of the last iteration of SGD
        w: model parameters as a numpy array of shape (D,) of the last iteration of SGD
    """

    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute a stochastic gradient and loss
            grad = grad_logistic(y_batch, tx_batch, w) + lambda_ * w
            w = w - gamma * grad
    
    loss = compute_loss(y, tx, w, "cross_entropy")

    return w, loss

def cross_validation_logreg(y, x, k_indices, k, gamma, max_iters, batch_size, lambda_):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with constant
    tx_tr = np.c_[np.ones(x_tr.shape[0]), x_tr]
    tx_te = np.c_[np.ones(x_te.shape[0]), x_te]
    # SGD
    w_initial = np.zeros(tx_tr.shape[1])

    w, loss_tr = stochastic_logistic_regression_reg(y_tr, tx_tr, w_initial, batch_size, max_iters, gamma, lambda_)
    # calculate the loss for train and test data
    loss_te = compute_loss(y_te, tx_te, w, "cross_entropy")
    return float(loss_tr), float(loss_te)

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
 

    cross_validation_visualization(gammas, rmse_tr, rmse_te)
    print(
        "For SGD, the choice of gamma which leads to the best test mse is %.5f with a test mse of %.3f"
        % (best_gamma, best_rmse)
    )
    return best_gamma, best_rmse