# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
from helpers import batch_iter
import numpy as np
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    
    error = y - np.dot(tx, w)
    
    return -tx.T.dot(error) / len(y)



def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
	ws = [initial_w]
	losses = []
	w = initial_w
	    
	for n_iter in range(max_iters):
	        
	    y_batch, tx_batch = batch_iter(y, tx, batch_size) #TODO check if code compatible with batch_iter
	        
	    gradient = compute_stoch_gradient(y_batch, tx_batch, w)
	    loss = compute_loss(y_batch, tx_batch, w)

	    w = w - gamma * gradient
	        
	    # store w and loss
	    ws.append(w)
	    losses.append(loss)
        
    return losses, ws