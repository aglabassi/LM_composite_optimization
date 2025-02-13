#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:12:31 2025

@author: aglabassi
"""

import numpy as np

from itertools import product
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from LM_composite_optimization.utils import plot_results_sensitivity
from matrix_utils import trial_execution_matrix

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





    
    
run = True
n=40
r_true = 2
r = 5
rel_init_start = 10**-3
rel_error_exp = 8
rel_epsilon_stop = 10**(-1*rel_error_exp)
tests = [ (5,100)] 
problems =  [ 'Burer-Monteiro Matrix Sensing'] #keep only one
methods = [ 'Subgradient descent', 'Gauss-Newton','Levenberg–Marquardt (ours)']
corruption_levels = [0]
qs = [ 0.95, 0.96, 0.97 ]
lambdas = [ 10**-5 ]
gammas = [10**-i for i in range(0,12)]
K  = 10
n_trial = 100
np.random.seed(42)

base_dir = os.path.join(repo_root, 'LM_composite_optimization/experiment_results/hyperparameter_sensitivity')
   

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
                            rel_init_start, K, 1, base_dir,
                            methods,
                            corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i], geom_decay=True
                        )
    
                    elif problem == 'Assymetric Matrix Sensing':
                        losses_ = trial_execution_matrix(
                            range(1), n, r_true, d, [(r_test, c)],
                            rel_init_start, K, 1, base_dir,
                            methods,
                            symmetric=False, corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i], geom_decay=True
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
            save_path = os.path.join(base_dir, f'to_be_plotted_{problem}_{corr_level}_{r_test}_{c}_{q}.pkl')            
            save(to_be_plotted[problem], save_path)

# Call the plotting function
font_size=30
for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):
    for problem, method  in product(problems, methods):
        to_be_plotted = load(os.path.join(base_dir, f'to_be_plotted_{problem}_{corr_level}_{r_test}_{c}_{q}.pkl'))     
        plot_results_sensitivity(to_be_plotted, corr_level, q, r_test, c, gammas, lambdas, font_size, rel_error_exp, problem, base_dir)
    
