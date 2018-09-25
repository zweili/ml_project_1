import numpy as np


def mse(error):
    """loss function using mean square error but multiply by 0.5
        to make the gradient calculation easier for other loss functions"""

    return 0.5 * np.mean(error ** 2)


def mae(error):
    """ loss function using mean absolute error --> more robust than mse"""

    return np.mean(abs(error))


def huber(error, delta):
    """loss function using Huber --> convex and robust but delta difficult to set"""

    if delta is None:
        delta = 0

    if abs(error) <= delta:

        return 0.5 * error ** 2

    else:
        return delta * abs(error) - 0.5 * delta ** 2
