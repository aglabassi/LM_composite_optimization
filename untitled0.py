#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:39:09 2024

@author: aglabassi
"""

import numpy as np
p = 10**-8

def c(x):
    return np.array( [x[i]**2 for i in range(0,len(x)-1) ]  + [ x[-1]**4  ]   )
    #return x**2


methods = [ 'Gradient descent', 'Levenberg–Marquardt (ours)']
n_iter = 10000
n = 10
r = 2
x_star = np.array( [ 1 ]*(r-1) + [ 1/100  ] + [ 0 ]* (n-r) )


errs =  dict()
z_star = c(x_star)




def nabla_c(x):
    
    return np.diag(  [2*x[i] for i in range(0,len(x) - 1) ]  + [ 4*x[-1]**3 ])
    #return np.diag( 2*x )

def h(z):
    #return 0.5*( (z[0] - z[1])**2 + z[1]**2 ) + np.linalg.norm() 
    return 0.5*np.linalg.norm(z - z_star)**2


def dh(z):
    x,y = z[0],z[1]
    #return np.array( [x-y, 2*y-x] + [ z[i] -1  for i in range(2,r)  ] + [ z[i]  for i in range(r,n)  ] )
    return z-z_star


import matplotlib.pyplot as plt

for method in methods:
    errs[method] = []
    x = x_star + p* np.ones(n)
    print(x)
    
    for i in range(n_iter):
        err = np.linalg.norm(x-x_star)
        print(err/(np.linalg.norm(x_star)))
        errs[method].append(err)
        
        v = dh(c(x))
        jacob_c = nabla_c(x)
        grad = jacob_c.T@v
        stepsize = (h(c(x)) - 0) /( (np.dot(grad,grad)) if method == 'Gradient descent' else np.dot(v, v) )
        damping = np.linalg.norm( c(x) - c(x_star)) if method ==  'Levenberg–Marquardt (ours)' else 0
        if method == 'Gradient descent':
            direction = grad
            
            
        else:
            direction,_,_,_= np.linalg.lstsq( jacob_c.T@jacob_c + damping*np.eye(n), grad, rcond = -1)
            
        
        x = x - stepsize*direction


    plt.plot(errs[method], label=method)
plt.yscale('log')
plt.legend()

        
        