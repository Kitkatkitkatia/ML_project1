from Code.costs import *
from Code.helpers import batch_iter

################ Gradient #####################################

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    grad = -(tx.T @ e)/len(e)
    return grad

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def grad_logistic(y, tx, w):
    pred = sigmoid(tx@w)
    grad = tx.T @ (pred - y) * (1 / len(pred))
    return grad
    #return (1/len(tx)) * (tx.T @ (-y * sigmoid(-y * (tx @ w))))


############### Gradient Descent ############################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape(D,), D is the number of features. The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: loss value (scalar) of the last iteration of SGD
        w: model parameters as a numpy array of shape (2, ) of the last iteration of SGD
    """

    w = initial_w
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w, "mse")

        w = w - gamma * grad
        
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss

################# Stochastic Gradient Descent ##################################

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma): # mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape(D,), D is the number of features. The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

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
            grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

            loss = compute_loss(y, tx, w, "mse")
            
        print(
            "SGD iter. {bi}/{ti}: loss={l}, norm_w ={norm_w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, norm_w = np.linalg.norm(w)
            )
        )
    return w, loss

################ Least Squares #############################

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss(y, tx, w, "mse")

################### Ridge Regression ########################

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: loss value (mean squared error by default)

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """

    N, D = tx.shape
    A = tx.T @ tx + 2*N*lambda_*np.eye(D)
    b = tx.T @ y

    w = np.linalg.solve(A, b)

    return w, compute_loss(y, tx, w, "mse")

################# Logistic Regression ######################

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic Regression using gradient descent.
    
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            initial_w: numpy array of shape(D,), D is the number of features. The initial guess (or the initialization) for the model parameters
            batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
            max_iters: a scalar denoting the total number of iterations of GD
            gamma: a scalar denoting the stepsize
    
        Returns:
            loss: loss value (scalar) of the last iteration of GD
            w: model parameters as a numpy array of shape (D,) of the last iteration of GD
        """

    w = initial_w
    loss = 0
    for n_iter in range(max_iters): 
    
        grad = grad_logistic(y, tx, w)
        
        w = w - gamma * grad

        loss = compute_loss(y, tx, w, "cross_entropy")
        print(
            "log reg iter. : loss={l}".format(l=loss)
        )
        
    return w, loss

################# Reg Logistic Regression #################

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized Logistic Regression using gradient descent.
    
        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N,D), D is the number of features.
            initial_w: numpy array of shape(D,), D is the number of features. The initial guess (or the initialization) for the model parameters
            batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
            max_iters: a scalar denoting the total number of iterations of GD
            gamma: a scalar denoting the stepsize
    
        Returns:
            loss: loss value (scalar) of the last iteration of GD
            w: model parameters as a numpy array of shape (D,) of the last iteration of GD
        """

    w = initial_w
    loss = 0
    for n_iter in range(max_iters): 
    
        grad = grad_logistic(y, tx, w) + lambda_ * w
        
        w = w - gamma * grad

        loss = compute_loss(y, tx, w, "cross_entropy")

        print(
            "log reg iter. : loss={l}".format(l=loss)
        )
        
    return w, loss