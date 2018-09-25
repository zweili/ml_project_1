import numpy as np
from loss_functions import mse


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """linear regression with gradient descent"""

    w = initial_w
    loss = 0

    for i in range(max_iters):
        # gradient descent in action + calculation of the loss
        error = y - np.dot(tx, w)
        gradient = np.dot(np.mean(error), -tx)
        loss = mse(error)

        # update the weights
        w -= gamma * gradient

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """linear regression with stochastic gradient descent"""

    w = initial_w
    loss = 0

    for i in range(max_iters):
        # gradient descent in action + calculation of the loss
        rint = np.random_integers(0, len(y))
        error = y - np.dot(tx, w)
        gradient = np.dot(np.mean(error[rint]), -tx[rint])
        loss = mse(error)

        # update the weights
        w -= gamma * gradient
    return w, loss


def least_squares(y, tx):
    """normal linear regression"""
    return


def ridge_regression(y, tx, lambda_):
    """linear regression with Tikhonov method (ridge)"""
    return


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """logistic regression"""
    return


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """"regularized logistic regression"""
    return
