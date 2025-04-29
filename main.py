#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:11:53 2025

@author: aglabassi
"""

import os
import torch
import pickle
from plotting_utils  import collect_compute_mean,\
 plot_losses_with_styles, plot_transition_heatmap, plot_results_sensitivity
from run_matrix_tensor_experiments import run_matrix_tensor_sensing_experiments
from run_hadamard_experiments import run_nonegative_least_squares


save_path = './'

# Set device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

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


def set_seed(seed=8):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(8)
torch.set_default_dtype(torch.float64)

# #Repeat until Gauss-Newton diverges. It will diverge at some point. Use polyak stepsize with gamma=1.
# n1 = 50
# n2=  50
# n3=10
# r_true = 2

# tensor = False  #Symmetric Matrix Factorization
# symmetric = True
# identity = True

# base_dir = os.path.join(save_path, 'experiment_results/polyak')
# os.makedirs(base_dir, exist_ok=True)
# loss_ord = 2 #L2 norm not squared
# initial_relative_error = 10**-2
# n_iter = 50

# # Experiment_setups: (overparameterization, condition number)
# experiment_setups = [(2,100)]

# methods = methods_test=  ['Polyak Subgradient', 'Levenberg-Marquardt (ours)']

# run_matrix_tensor_sensing_experiments(methods_test, experiment_setups, n1, n2, n3, r_true, 0, identity, device,
#         n_iter, base_dir, loss_ord, initial_relative_error, symmetric,
#         tensor=tensor, gamma_custom=1)

# errs, stds = collect_compute_mean(experiment_setups, loss_ord, r_true, False, methods,
#                               f'{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}', base_dir
#                                 )

# plot_losses_with_styles(errs, stds, r_true, loss_ord, base_dir,
#                       (('Symmetric ' if symmetric else 'Asymmetric ') +
#                       ('Tensor' if tensor else 'Matrix')), 1, intro_plot=False)


base_dir = os.path.join(save_path, 'experiment_results/polyak')
r_true = 10
r=100

kappa = 1
experiment_setups = [(r_true,1),(r,100)]
n_trial = 1
methods_test = methods = ['Polyak Subgradient', 'Levenberg-Marquardt (ours)', 'Gauss-Newton']
n_iter = 100
run_nonegative_least_squares(methods_test, experiment_setups, r_true,
                 1, device,
                 n_iter, base_dir, 1)

errs, stds = collect_compute_mean(experiment_setups, 2, r_true, False, methods,
                              'had', base_dir)

plot_losses_with_styles(errs, stds, r_true, 2, base_dir,
                      'Hadamard', 1, had=True)