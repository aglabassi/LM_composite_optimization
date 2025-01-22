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


def plot_results(to_be_plotted, corr_level, q, r_test, c, xy_axis_max, gammas, lambdas, font_size, rel_error_exp):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#785ef0'] * len(to_be_plotted)

    Xvals = np.arange(0, xy_axis_max)
    Yvals = np.arange(0, xy_axis_max)
    Xgrid, Ygrid = np.meshgrid(Xvals, Yvals)

    legend_elements = []

    for idx, (problem, data) in enumerate(to_be_plotted.items()):
        median_grid = np.zeros((xy_axis_max, xy_axis_max))

        for i in range(xy_axis_max):
            for j in range(xy_axis_max):
                median_val = data[i, j][0]  # (median, (5th,95th))
                median_grid[i, j] = median_val

                ax.scatter(
                    Xgrid[i, j],
                    Ygrid[i, j],
                    median_val,
                    color=colors[idx],
                    s=40
                )

        surf = ax.plot_surface(
            Xgrid, Ygrid, median_grid,
            color=colors[idx],
            alpha=0.6,
            edgecolor='none'
        )

        legend_elements.append(Patch(facecolor=colors[idx], label=problem))

    ax.set_xticks(Xvals)
    ax.set_yticks(Yvals)
    ax.zaxis.set_tick_params(labelsize=font_size // 2)
    ax.set_xticklabels([fr"$10^{{-{i}}}$" for i in Xvals], fontsize=font_size // 2)
    ax.set_yticklabels([fr"$10^{{-{j}}}$" for j in Yvals], fontsize=font_size // 2)

    ax.set_xlabel(r"$\lambda$", fontsize=font_size, labelpad=20)
    ax.set_ylabel(r"$\gamma$", fontsize=font_size, labelpad=20)
    ax.set_xlim(max(Xvals), min(Xvals))

    bound = f"10^{{-{rel_error_exp}}}"
    ax.set_zlabel(
        rf"Iterations for $\frac{{\|z_k - z^*\|_2}}{{\|z_k\|_2}} \leq {bound}$",
        fontsize=font_size,
        labelpad=30,
    )

    ax.set_title(f"Decay Parameter q={q}", fontsize=font_size)

    plt.tight_layout()
    base_dir = './exp2'
    save_path = os.path.join(base_dir, f"plot_{q}_{corr_level}_{r_test}_{c}.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"Figure saved to: {save_path}")
    plt.show()
    
    
run = False
n=10
r_true = 2
r = 5
rel_init_start = 10**-3
rel_error_exp = 8
rel_epsilon_stop = 10**(-1*rel_error_exp)
tests = [ (r_true,1), (r,10)] 
problems =  [ 'Burer-Monteiro Matrix Sensing']
corruption_levels = [0]
qs = [0.97, 0.98, 0.99]
xy_axis_max = 2
lambdas = [ 10**(-i) for i in range(0,xy_axis_max )]
gammas = [ 10**(-i) for i in range(0,xy_axis_max )]
K  = 2
n_trial = 1


if run:
    for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):
        d = 10 * n * r_true if r_test == r_true else 20 * n * r_true
        to_be_plotted = {problem: np.zeros((xy_axis_max, xy_axis_max), dtype=object) for problem in problems}
    
        for problem in problems:
            for i, j in product(range(xy_axis_max), range(xy_axis_max)):
                last_indexes = []
    
                for _ in range(n_trial):
                    # Generate losses depending on the problem
                    if problem == 'Burer-Monteiro Matrix Sensing':
                        losses = trial_execution_matrix(
                            range(1), n, r_true, d, [(r_test, c)],
                            rel_init_start, K, 1, "./",
                            ['Levenberg–Marquardt (ours)'],
                            corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )['Levenberg–Marquardt (ours)']
    
                    elif problem == 'Assymetric Matrix Sensing':
                        losses = trial_execution_matrix(
                            range(1), n, r_true, d, [(r_test, c)],
                            rel_init_start, K, 1, "./",
                            ['Levenberg–Marquardt (ours)'],
                            symmetric=False, corr_factor=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )['Levenberg–Marquardt (ours)']
    
                    elif problem == 'Symmetric Tensor Sensing':
                        losses = run_methods(
                            ['Levenberg–Marquardt (ours)'],
                            [(r_test, c)], n, r_true, d, False, 'cpu',
                            K, False, './', 1, rel_init_start,
                            corr_level=corr_level, q=q,
                            gamma=gammas[j], lambda_=lambdas[i]
                        )['Levenberg–Marquardt (ours)']
    
                    # Find the first iteration index where losses < rel_epsilon_stop
                    idx = np.where(np.array(losses) < rel_epsilon_stop)[0]
                    last_index = idx[0] if len(idx) > 0 else K
                    last_indexes.append(last_index)
    
                # Calculate median and percentiles
                median_last_index = np.median(last_indexes)
                shaded = np.percentile(last_indexes, [5, 95])
                to_be_plotted[problem][i, j] = (median_last_index, tuple(shaded))
            save(to_be_plotted, f'exp2/to_be_plotted_{problem}_{corr_level}_{r_test}_{c}.pkl')

# Call the plotting function

for corr_level, (r_test, c), q in product(corruption_levels, tests, qs):
    for problem  in problems:
        to_be_plotted = load(f'exp2/to_be_plotted_{problem}_{corr_level}_{r_test}_{c}.pkl')     
        plot_results(to_be_plotted, corr_level, q, r_test, c, xy_axis_max, gammas, lambdas, font_size, rel_error_exp)
    
