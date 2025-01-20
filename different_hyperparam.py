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

n=50
r_true = 2
r = 5
d = 10*n*r_true
rel_init_start = 10**-3
rel_error_exp = 10
rel_epsilon_stop = 10**(-1*rel_error_exp)
tests = [ (r_true,1)] 
problems =  ['symmetric matrix sensing']
corruption_levels = [0]
qs = [0.85]
xy_axis_max = 2 
lambdas = [ 10**(-i) for i in range(1,xy_axis_max + 1)]
gammas = [ 10**(-i*5) for i in range(1,xy_axis_max + 1)]
K  = 500
n_trial = 4

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl


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
                last_index = idx[0] if len(idx) > 0 else K
                last_indexes.append(last_index)

            median_last_index = np.median(last_indexes)
            # 5th and 95th percentiles
            shaded = np.percentile(last_indexes, [5, 95])
            to_be_plotted[problem][i, j] = (median_last_index, tuple(shaded))

    # ------------------- PLOTTING BLOCK BEGINS HERE --------------------
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare distinct colors if you have multiple "problems"
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    font_size = 30

    # LaTeX / font settings
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size

    # Build integer coordinate grids (1..xy_axis_max)
    Xvals = np.arange(1, xy_axis_max + 1)
    Yvals = np.arange(1, xy_axis_max + 1)
    Xgrid, Ygrid = np.meshgrid(Xvals, Yvals)

    legend_elements = []

    for idx, problem in enumerate(problems):
        median_grid = np.zeros((xy_axis_max, xy_axis_max))

        # Fill median_grid and plot scatter
        for i in range(xy_axis_max):
            for j in range(xy_axis_max):
                median_val = to_be_plotted[problem][i, j][0]  # (median, (5th,95th))

                # Store in a 2D array, so we can plot a surface
                median_grid[i, j] = median_val

                # Xgrid[i,j] = i+1, Ygrid[i,j] = j+1; scatter that point
                ax.scatter(
                    Xgrid[i, j],
                    Ygrid[i, j],
                    median_val,
                    color=colors[idx],
                    s=40  # marker size
                )

        # Plot a surface from the median values
        surf = ax.plot_surface(
            Xgrid,
            Ygrid,
            median_grid,
            color=colors[idx],
            alpha=0.6,
            edgecolor='none'
        )

        # For legend
        legend_elements.append(Patch(facecolor=colors[idx], label=problem))

    # Set integer ticks along X, Y
    ax.set_xticks(Xvals)
    ax.set_yticks(Yvals)
    
    # Now set the custom LaTeX labels for each tick
    ax.set_xticklabels([fr"$10^{{-{i}}}$" for i in Xvals])
    ax.set_yticklabels([fr"$10^{{-{j}}}$" for j in Yvals])

    # Label axes with extra padding to avoid overlap
    ax.set_xlabel(r"$\lambda$",
                  fontsize=font_size,
                  labelpad=20)
    ax.set_ylabel(r"$\gamma$",
                  fontsize=font_size,
                  labelpad=20)

    # For the z-label, including your epsilon expression:
    bound = f"\\small{{10^{{-{rel_error_exp}}}}}"
    ax.set_zlabel(
       rf"Iterations for $\frac{{\|z_k - z^*\|_2}}{{\|z_k\|_2}} \leq {bound}$",
        fontsize=font_size,
        labelpad=30
    )

    # Title and legend
    title_str = f"corr={corr_level}, r_test={r_test}, c={c}, q={q}"
    ax.set_title(title_str, fontsize=font_size)

    ax.legend(handles=legend_elements, loc='upper right')

    # Optionally tighten layout
    plt.tight_layout()
    plt.show()