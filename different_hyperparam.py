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

n=10
r_true = 2
r = 5
d = 13*n*r_true
rel_init_start = 10**-3
rel_error_exp = 10
rel_epsilon_stop = 10**rel_error_exp
tests = [ (r_true,1)] 
problems =  ['symmetric matrix sensing']
corruption_levels = [0]
qs = [0.85,0.9, 0.95]
xy_axis_max = 10
lambdas = [ 10**-i for i in range(1,xy_axis_max + 1)]
gammas = [ 10**-i for i in range(1,xy_axis_max + 1)]
K  = 1000
n_trial = 5

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# ------------------------------------------------------------------------
# 3. MAIN LOOP
# ------------------------------------------------------------------------
for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):

    # Dictionary that will hold, for each problem, a 2D array with shape (xy_axis_max, xy_axis_max).
    # Each element is a tuple (median_last_index, (5th_percentile, 95th_percentile))
    to_be_plotted = {problem: np.zeros((xy_axis_max, xy_axis_max), dtype=object)
                     for problem in problems}
    #Populate
    for problem in problems:
        for i, j in product(range(xy_axis_max), range(xy_axis_max)):
            last_indexes = []
            for _ in range(n_trial):
                # Generate losses depending on the problem
                if problem == 'symmetric matrix sensing':
                    losses = trial_execution_matrix(
                        range(1), n, r_true, d, [(r_test, c)], 
                        rel_init_start, K, 1, "./", 
                        ['Levenberg–Marquardt (ours)'],
                        corr_factor=corr_level, q=q, 
                        gamma=gammas[j], lambda_=lambdas[i]
                    )['Levenberg–Marquardt (ours)']

                elif problem == 'assymetric matrix sensing':
                    losses = trial_execution_matrix(
                        range(1), n, r_true, d, [(r_test, c)], 
                        rel_init_start, K, 1, "./", 
                        ['Levenberg–Marquardt (ours)'],
                        symmetric=False, corr_factor=corr_level, q=q, 
                        gamma=gammas[j], lambda_=lambdas[i]
                    )['Levenberg–Marquardt (ours)']

                elif problem == 'symmetric tensor sensing':
                    losses = run_methods(
                        ['Levenberg–Marquardt (ours)'], 
                        [(r_test, c)], n, r_true, d, False, 'cpu', 
                        K, False, './', 1, rel_init_start,
                        corr_level=corr_level, q=q, 
                        gamma=gammas[j], lambda_=lambdas[i]
                    )['Levenberg–Marquardt (ours)']

                # Find first iteration index where losses < rel_epsilon_stop
                # If none is found, set it to K
                losses = np.array(losses)
                idx = np.where(losses < rel_epsilon_stop)[0]
                last_index = idx[-1] if len(idx) > 0 else K
                last_indexes.append(last_index)

            median_last_index = np.median(last_indexes)
            # 5th and 95th percentiles
            shaded = np.percentile(last_indexes, [5, 95])
            to_be_plotted[problem][i, j] = (median_last_index, tuple(shaded))

    #Plot
    # Create a single 3D figure for this combination of (corr_level, r_test, c, q)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare a colormap or list of distinct colors, one per problem.
    # Example: just define some colors manually. Adjust to match # of problems.
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    # We'll need something to build the legend entries. 
    legend_elements = []
    
    # Build a meshgrid for the XY plane: shape = (xy_axis_max, xy_axis_max)
    X = np.arange(xy_axis_max)
    Y = np.arange(xy_axis_max)
    X, Y = np.meshgrid(X, Y)
    
    # Plot surfaces for all problems in the same figure:
    for idx, problem in enumerate(problems):
        # Prepare median data for the current problem
        median_grid = np.zeros((xy_axis_max, xy_axis_max))
    
        for i in range(xy_axis_max):
            for j in range(xy_axis_max):
                # Recall: to_be_plotted[problem][i, j] = (median_val, (5th, 95th))
                median_val = to_be_plotted[problem][i, j][0]
                median_grid[i, j] = median_val
    
        # Plot the median surface for this problem in 3D
        # We set a single color, ignoring the data-based colormap for clarity.
        surf = ax.plot_surface(X, Y, median_grid,
                               color=colors[idx],
                               alpha=0.6,
                               edgecolor='none')
    
        # Create a legend patch that matches the color
        legend_patch = Patch(facecolor=colors[idx], label=problem)
        legend_elements.append(legend_patch)
    
    # Axes labels, title, legend
    ax.set_xlabel(r"$i$ (for $\lambda = 10^{-i}$)")
    ax.set_ylabel(r"$j$ (for $\gamma = 10^{-j}$)")
    ax.set_zlabel(fr"Iterations to reach relative error $\epsilon = 10^{{-{rel_error_exp}}}$")
    title_str = f"corr={corr_level}, r_test={r_test}, c={c}, q={q}"
    ax.set_title(title_str)
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
     
                    
        
