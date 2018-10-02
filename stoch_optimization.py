from math import sqrt
from helpers import batch_iter
import numpy as np
from costs import compute_loss
from stochastic_gradient_descent import compute_stoch_gradient

def adaGrad(y, tx, initial_w, batch_size, max_iters, gamma): #optimization of SDG
    """adaptive gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    epsilon = 1e-8 #to avoid zero division
    Gti = np.zeros(initial_w.shape[0])
    
    for n_iter in range(max_iters):

        y_batch, tx_batch = batch_iter(y, tx, batch_size) #TODO check if code compatible with batch_iter
        
        gradient = compute_stoch_gradient(y_batch, tx_batch, w)
        loss = compute_loss(y_batch, tx_batch, w)
        
        Gti += gradient**2

        adjusted_gamma = gamma / (epsilon + np.sqrt(Gti))

        w = w - adjusted_gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
   
    return losses, ws


def adam(y, tx, initial_w, batch_size, max_iters, gamma): #Better optimization of SGD
    """Adam gradient descent algorithm."""
    beta1 = 0.9  # momentum
    beta2 = 0.999 # RMSprop
    epsilon = 1e-8 # to avoid zero division
    t, v, b = 0, 0, 0
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        
        t +=1
        y_batch, tx_batch = batch_iter(y, tx, batch_size) #TODO check if code compatible with batch_iter
        gradient = compute_stoch_gradient(y_batch, tx_batch, w)
        loss = compute_loss(y_batch, tx_batch, w)
        
        v = v * beta1 + (1 - beta1) * gradient 
        b = b * beta2 + (1 - beta2) * gradient ** 2
        
        v_bias_corr = v / (1 - beta1 ** t)
        b_bias_corr = b / (1 - beta2 ** t)
        
        div = gamma * v_bias_corr / (np.sqrt(b_bias_corr) + epsilon)
        
        w = w - div
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return losses, ws