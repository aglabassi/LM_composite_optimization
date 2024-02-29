#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:25:35 2024

@author: aglabassi
"""


import numpy as np
from scipy.optimize import minimize

def nabla_pinv_c_v(X, V):
    def objective(G):
        G = G.reshape(X.shape)  # Reshape G back to the shape of X for matrix operations
        term = np.dot(G, X.T) + np.dot(X, G.T) - V
        return np.linalg.norm(term, 'fro')**2

    G_init = np.random.rand(*X.shape).flatten()  # Flatten G for the optimizer

    def gradient(G_flat):
        G = G_flat.reshape(X.shape)
        # Compute the gradient based on the derived formula
        grad = 2 * (np.dot(G, np.dot(X.T, X)) + np.dot(np.dot(X, X.T), G) - np.dot(V, X) - np.dot(X.T, V))
        return grad.flatten()


    # Use the 'minimize' function without explicitly specifying the 'jac' (gradient)
    # SciPy will approximate the gradient numerically if 'jac' is not provided
    res = minimize(fun=objective, x0=G_init, method='CG', jac=gradient)

    G_opt = res.x.reshape(X.shape)  # Reshape the optimized G back to the shape of X
 
    return G_opt

def run_gnp(X0, loss_ord, n_iter, A,A_t, y):
#A and A_t are linear functions mapping respectively matrix nxn to vector d and vector d to matrix nxn
    X = X0
    losses = []
    for k in range(n_iter):

        
        c_X = X@X.T
        A_c_X = A(c_X)
        h_c_X = np.linalg.norm(A_c_X - y, ord=loss_ord)**loss_ord
        losses.append(h_c_X)
        
                
        print('Iteration number: ', k)
        print('h(c(x)) =', h_c_X)
        print('---------------------')
        
        if loss_ord == 1:
            V = A_t(np.sign(A_c_X - y))
        elif loss_ord == 2:
            V = A_t((A_c_X - y))
        else:
            raise NotImplementedError
        
        update_direction = nabla_pinv_c_v(X, V)
        projected_v_length = np.sum(np.multiply(X@update_direction.T @X + update_direction@(X.T@X), update_direction))
        stepsize = h_c_X/projected_v_length
        
        X = X - stepsize*update_direction
        
        
        
        
        
    return losses
        
        
        
        
        