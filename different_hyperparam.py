#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:12:31 2025

@author: aglabassi
"""

import numpy as np
from utils import trial_execution_matrix
from main_tensor_symmetric import run_methods
np.random.seed(42)
from itertools import product
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import pickle  

def save(obj, filename):
    """Saves an object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")

def load(filename):
    """Loads an object from a file using pickle."""
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filename}")
    return obj



# LaTeX / font settings
font_size = 30
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = font_size


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def plot_results(to_be_plotted_, corr_level, q, r_test, c, gammas, lambdas, font_size, rel_error_exp, problem):
   
    method_colors = {
        'Subgradient descent': '#dc267f',
        'Gradient descent': '#dc267f',  # Same color as 'Subgradient descent'
        'Scaled gradient': '#ffb000',
        'Scaled gradient($\lambda=10^{-3}$)': '#ffa800',
        'Scaled gradient($\lambda=10^{-8}$)': '#CD853F',
        'Scaled subgradient': '#ffaf00',  # Same color as 'Scaled gradient'
        'OPSA($\lambda=10^{-3}$)': '#97e60d',
        'OPSA($\lambda=10^{-8}$)': '#94cc1a',
        'Precond. gradient': '#fe6100',
        'Gauss-Newton': '#648fff',
        'Levenberg–Marquardt (ours)': '#785ef0'
    }
    X_val = [10**-i for i in range(len(gammas))]
    base_dir = './exp2'
    os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists
    
    for i, lambda_ in enumerate(lambdas):
        plt.figure(figsize=(10, 6))  # Create a new figure for each lambda
        for method in to_be_plotted.keys():
            color = method_colors[method]
            data = [to_be_plotted[method][i, j][0] for j in range(len(gammas))]
            noise  = [to_be_plotted[method][i, j][1] for j in range(len(gammas)) ]  # Assume noise is a tuple (low, high)
    
            # Compute the noise bounds
            noise_low = [noise[j][0] for j in range(len(gammas))]
            noise_high = [noise[j][1] for j in range(len(gammas))]
    
            # Plot scatter points, connecting lines, and noise shading
            plt.plot(X_val, data, label=method, color=color, linestyle='-', linewidth=2, marker='o')
            plt.fill_between(X_val, noise_low, noise_high, color=color, alpha=0.2)
    
        # Set labels and title with LaTeX rendering
        plt.xlabel(r"$\gamma$", fontsize=font_size)
        bound = 10
        plt.ylabel(rf"Iterations for $\frac{{\|z_k - z^*\|_2}}{{\|z_k\|_2}} \leq 10^{- rel_error_exp}$", fontsize=font_size)
        plt.title(f"$q={q}, \lambda = {lambda_}$", fontsize=font_size)
        plt.xscale('log')
        
        # Customize ticks and add grid
        plt.xticks(fontsize=font_size//2)
        plt.yticks(fontsize=font_size//2)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=font_size//2)
    
        # Save the plot to a file
        save_path = os.path.join(base_dir, f"plot_{problem}_{q}_{corr_level}_{r_test}_{c}_{lambda_}.pdf")
        plt.savefig(save_path, format='pdf')
        print(f"Figure saved to: {save_path}")
        plt.show()
        plt.close()  # Close the figure to free memory
        
    



    
    
run = True
n=40
r_true = 2
r = 5
rel_init_start = 10**-3
rel_error_exp = 8
rel_epsilon_stop = 10**(-1*rel_error_exp)
tests = [ (r_true,1)] 
problems =  [ 'Burer-Monteiro Matrix Sensing'] #keep only one
methods = ['Subgradient descent', 'Levenberg–Marquardt (ours)']
corruption_levels = [0]
qs = [0.98]
lambdas = [ 10**(-5)]
gammas = [ 10**(-i) for i in range(0,11 )]
K  = 1000
n_trial = 10


if run:
    for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):
        d = 20 * n * r_true
        to_be_plotted = dict( (problem, dict( (method,np.zeros((len(lambdas), len(gammas)), dtype=object)) for method in methods)) for problem in problems )
    
        for problem in problems:
            for i, j in product(range(len(lambdas)), range(len(gammas))):
                last_indexes = dict(  (method, [] ) for method in methods)
              
                for _ in range(n_trial):
                    # Generate losses depending on the problem
                    if problem == 'Burer-Monteiro Matrix Sensing':
                        losses_ = trial_execution_matrix(
                            range(1), n, r_true, d, [(r_test, c)],
                            rel_init_start, K, 1, "./",
                            methods,
                            corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )
    
                    elif problem == 'Assymetric Matrix Sensing':
                        losses_ = trial_execution_matrix(
                            range(1), n, r_true, d, [(r_test, c)],
                            rel_init_start, K, 1, "./",
                            methods,
                            symmetric=False, corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )
    
                    elif problem == 'Symmetric Tensor Sensing':
                        losses_ = run_methods(
                            methods,
                            [(r_test, c)], n, r_true, d, False, 'cpu',
                            K, False, './', 1, rel_init_start,
                            corr_level=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )
                    
                    for method in methods:
                        losses = losses_[method]
                        # Find the first iteration index where losses < rel_epsilon_stop
                        idx = np.where(np.array(losses) < rel_epsilon_stop)[0]
                        last_index = idx[0] if len(idx) > 0 else K
                        last_indexes[method].append(last_index)
                        
                for method in methods:
                    median_last_index = np.median(last_indexes[method])
                    shaded = np.percentile(last_indexes[method], [5, 95])
                    to_be_plotted[problem][method][i, j] = (median_last_index, tuple(shaded))
                        
            save(to_be_plotted[problem], f'exp2/to_be_plotted_{problem}_{corr_level}_{r_test}_{c}.pkl')

# Call the plotting function

for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):
    for problem, method  in product(problems, methods):
        to_be_plotted = load(f'exp2/to_be_plotted_{problem}_{corr_level}_{r_test}_{c}.pkl')     
        plot_results(to_be_plotted, corr_level, q, r_test, c, gammas, lambdas, font_size, rel_error_exp, problem)
    
