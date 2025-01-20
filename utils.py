#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:53:34 2024

@author: aglabassi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import os
from scipy.linalg import svd
import glob

def collect_compute_mean(keys, loss_ord, r_true, res, methods, problem):
    losses = dict(( method, dict()) for method in methods )
    stds = dict( (method, dict()) for method in methods)

    for rank, cond_number in keys:
        for method in methods:
            file_pattern = f"experiments/{('res' if res else 'exp') + problem}_{method}_l_{loss_ord}_r*={r_true}_r={rank}_condn={cond_number}_trial_*.csv"
            file_list = glob.glob(file_pattern)
            data_list = []
            
            # Read each file and append its data to the data_list
            for file in file_list:
                data_list.append(np.loadtxt(file, delimiter=','))  # Assume the CSV is correctly formatted for np.loadtxt
            
            # Convert the list of arrays into a 2D numpy array for easier manipulation
            data_array = np.array(data_list)
            
            # Compute the mean across all trials (rows) for each experiment
            mean_values = np.mean(data_array, axis=0)
            std = np.std(data_array, axis=0)
            
            losses[method][(rank, cond_number)]  = mean_values
            stds[method][(rank, cond_number )]  = std
   
    return losses, stds



def trial_execution_matrix(trials, n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric=True, identity=False, corr_factor=0,
                           gamma=0.00001, lambda_=0.1, q=0.8):
    
    for trial in trials:
        A, A_adj = create_rip_transform(n, d, identity)
    
        for r, cond_number in keys:
            X_true = generate_matrix_with_condition(n, r_true, cond_number)
            Y_true = generate_matrix_with_condition(n, r_true, cond_number)
            
            if symmetric:
                M_true =  X_true @ X_true.T
            else:
                M_true = X_true @ Y_true.T
                
                
                
            y_true = A(M_true)
            num_ones = int(y_true.shape[0]*corr_factor)
            mask_indices = np.random.choice(y_true.shape[0], size=num_ones, replace=False)
            mask = np.zeros(y_true.shape[0])
            mask[mask_indices] = 1

            y_true = y_true + np.linalg.norm(y_true)*np.random.normal(size=y_true.shape[0])*mask
  
     
  
            pert = np.random.rand(*M_true.shape)
            if symmetric:
                pert = (pert + pert.T) /2
            Z0 = M_true + (init_radius_ratio* np.linalg.norm(M_true)/ (np.linalg.norm(pert))) * pert

    
            U, Sigma, VT = np.linalg.svd(Z0)
            U_r = U[:, :r]
            Sigma_r = Sigma[:r]
            VT_r = VT[:r, :]
            
            # Compute Sigma_r^{1/2}
            Sigma_r_sqrt = np.sqrt(Sigma_r)
            
            # Compute X and Y
            X0 = U_r * Sigma_r_sqrt
            Y0 = (VT_r.T) * Sigma_r_sqrt
            outputs = dict()
            for method in methods:
                if symmetric:
                    losses = matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, 
                                             r_true, cond_number, method, base_dir, trial, gamma_init=gamma, 
                                             damping_init=lambda_, q=q)
                else:
                    losses = matrix_recovery_assymetric(X0, Y0, X_true, Y_true, M_true, T, A, A_adj, y_true, loss_ord, 
                                                        r_true, cond_number, method, base_dir, trial, gamma_init=gamma, 
                                                        damping_init=lambda_, q=q)
                outputs[method]= losses 
    return outputs





import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from matplotlib.lines import Line2D
def plot_losses_with_styles(losses, stds, r_true, loss_ord, base_dir, problem, kappa, num_dots=0.1):
    """
    Plots the losses with distinct styles for methods, parameterizations, and ill-conditioning levels.

    Parameters:
    - losses (dict): Nested dictionary containing loss values.
    - stds (dict): Nested dictionary containing standard deviations of losses.
    - r_true (float): The true parameterization value for comparison.
    - loss_ord (int): An identifier for the loss order.
    - base_dir (str): Base directory to save the plot.
    - problem (str): Type of problem (e.g., 'Burer-Monteiro', 'Asymmetric', etc.).
    - kappa (float): Ill-conditioning parameter.
    - num_dots (float): Fraction to determine the number of points to plot.
    """

    # Enable LaTeX rendering if desired
    mpl.rcParams['text.usetex'] = True  # Set to True if LaTeX is installed and desired

    # Adjust matplotlib parameters for publication quality
    mpl.rcParams['figure.figsize'] = (12, 9)  # Increased size for better visibility
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math rendering
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 10



    colors = dict()
    
    methods_ = list(losses.keys())
    
    for m in methods_:
        for m2 in methods_:
            assert losses[m].keys() == losses[m2].keys(), "All methods must have the same keys."



    markers = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']  # Extended markers list
    linestyles = ['-', ':']  # Different linestyles

    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=25)

    keys = list(losses[methods_[0]].keys())

    rs = []
    cs = []
    for k in keys:
        r, c = k
        if r not in rs:
            rs.append(r)
        if c not in cs:
            cs.append(c)

    # Machine epsilon for float64
    epsilon_machine = np.finfo(float).eps

    # Initialize dictionaries to store unique styles for legends
    marker_styles = {}
    linestyle_styles = {}
    
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
    'Gauss-Newton': '#648fff'  ,
    'Levenberg–Marquardt (ours)': '#785ef0'
    }

    methods = [ method for idx_m, (method, color) in enumerate(method_colors.items(), start=1)   if method in methods_ ]
    for idx_, method in enumerate(methods):

        color =  method_colors[method] 

        
        for idx, k in enumerate(keys):
            r_index = rs.index(k[0])
            c_index = cs.index(k[1])
            errs = np.array(losses[method][k])
            std = np.array(stds[method][k])

            # Determine the index where errors have converged to machine epsilon
            convergence_threshold = 1e-12   # Slightly above machine epsilon to account for numerical errors
            converged_indices = np.where(errs <= convergence_threshold)[0]
            print(method, k)
            if converged_indices.size > 0:
                last_index = converged_indices[0] + 1  # Include the converged point
            else:
                last_index = len(errs)

            # Slice the errors up to the convergence point
            errs = errs[:last_index]
            std = std[:last_index]

            num_dots_adapted = int(num_dots * last_index)
            if k[1] > 1:
                num_dots_adapted //=2

            start = 0
            num_dots_adapted = max(1, num_dots_adapted)
            indices = np.arange(start, last_index, num_dots_adapted)
            if k[1] > 1:
                indices = [ i for idx,i in enumerate(indices) if idx %2 != 0]

            indices = np.hstack((np.zeros(1, dtype=int), indices))
            if indices.size == 0:
                indices = np.array([0])  # Ensure at least one point is plotted
            elif indices[-1] != last_index - 1:
                # Include the last index to reach the convergence threshold
                indices = np.append(indices, last_index - 1)

            # Determine parameterization (overparam or exact param)
            over_exact_label = "Overparameterized" if k[0] > r_true else "Exact parameterization"
            if over_exact_label not in linestyle_styles:
                linestyle = linestyles[1 if k[0] > r_true else 0]
                linestyle_styles[over_exact_label] = linestyle  # Store linestyle for the parameterization
            else:
                linestyle = linestyle_styles[over_exact_label]

            # Determine marker based on ill-conditioning (kappa value)
            kappa_label = "Ill-conditioned" if k[1] > 1 else "Well-conditioned"
            if kappa_label not in marker_styles:
                marker = markers[1 if k[1] > 1 else 0]
                marker_styles[kappa_label] = marker  # Store marker for the kappa value
            else:
                marker = marker_styles[kappa_label]

            # Plotting the mean errors
            if errs[-1]  == 1:#divergent method
                diverged_idx = np.where(errs == errs[-1] )[0][0]
                tmp = np.where(diverged_idx < indices)[0][0]
                indices = indices[:tmp+1]
                errs *= (idx_ + 1)
                
            print(indices)
            ax.plot(
                indices,
                errs[indices],
                linestyle=linestyle,
                color=color,
                marker=marker,
                label=None  # Labels are handled separately
            )

            # Fill between for error bands
            ax.fill_between(
                indices,
                errs[indices] - std[indices],
                errs[indices] + std[indices],
                alpha=0.2,
                color=color
            )

    # Set the plot title based on the problem type
    object_ = {
        'Burer-Monteiro': 'Matrix',
        'Asymmetric': 'Matrix',
        'Hadamard': 'Square Vector',
        'Tensor': 'Tensors'
    }.get(problem, '')

    # Setting labels with LaTeX formatting
    ax.set_xlabel('Iteration', fontsize=30)
    ax.set_ylabel(r'Relative Distance $\frac{\left\| z_k - z^\ast \right\|_2}{\left\| z^\ast \right\|_2}$', fontsize=30)
    ax.set_yscale('log')

    # Adding grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjusting tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1, labelsize=26)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.5, labelsize=26)

    # Create custom legend handles
    # Methods legend (colors only)
    method_handles = [
        Line2D([0], [0], color=method_colors[method], lw=2) for method in methods
    ]
    method_labels = methods

    # Markers legend (parameterization), markers in black
    marker_handles = [
        Line2D([0], [0], marker=marker_styles[label], color='black', linestyle='None', markersize=8)
        for label in marker_styles
    ]
    marker_labels = list(marker_styles.keys())

    # Linestyles legend (ill-conditioning), linestyles in black
    linestyle_handles = [
        Line2D([0], [0], linestyle=linestyle_styles[label], color='black', lw=2)
        for label in linestyle_styles
    ]
    linestyle_labels = list(linestyle_styles.keys())
    # Combined legend for markers and linestyles at upper right
    combined_handles = marker_handles + linestyle_handles
    combined_labels = marker_labels + linestyle_labels

    # Determine the number of entries in each legend
    num_legend1 = len(method_handles)
    num_legend2 = len(marker_handles) + len(linestyle_handles)

    # Find the maximum number of entries
    max_entries = max(num_legend1, num_legend2)

    # Function to create a dummy handle
    def create_dummy_handle():
        return Line2D([0], [0], color='white', lw=0, markersize=0, label='_nolegend_')

    # Add dummy handles to legend1 if it's shorter
    if num_legend1 < max_entries:
        num_dummies = max_entries - num_legend1
        for _ in range(num_dummies):
            method_handles.append(create_dummy_handle())
            method_labels.append('')  # Empty label for dummy handle

    # Add dummy handles to legend2 if it's shorter
    if num_legend2 < max_entries:
        num_dummies = max_entries - num_legend2
        for _ in range(num_dummies):
            combined_handles.append(create_dummy_handle())
            combined_labels.append('')  # Empty label for dummy handle

    # Place legends
    if  (problem == 'Tensor' and loss_ord == 1):
        height = 0.61 
    elif problem  == 'Hadamard':
        height = 0.98
    elif (problem == 'CP' and loss_ord == 2):
        height = 1
    elif problem == 'Burer-Monteiro':
        height = 0.86
    elif problem == 'Asymmetric':
        height = 0.83
    else:
        height = 0.98

    # Methods legend at upper right
    legend1 = ax.legend(
        method_handles,
        method_labels,
        title='Methods',
        loc='upper right',
        bbox_to_anchor=(0.67, height),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=18,          # Set the font size of the legend labels to 18
        title_fontsize=20     # Set the font size of the legend title to 20
    )



    legend2 = ax.legend(
        combined_handles,
        combined_labels,
        title='Setting',
        loc='upper right',
        bbox_to_anchor=(1, height),  # Adjust the y-coordinate as needed
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=18,          # Set the font size of the legend labels to 18
        title_fontsize=20     # Set the font size of the legend title to 20
    )

    # Add the first legend back to the axes
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Tight layout for better spacing
    plt.tight_layout()

    # Saving the figure in a vector format (PDF)
    fig_path = os.path.join(base_dir, f'experiments/exp_l{loss_ord}_{problem}.pdf')
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()
    
def plot_transition_heatmap(
    success_matrixes: dict,
    d_trials: list,
    n: int,
    base_dir: str,
    problem: str = 'TransitionPlot',
    max_corr:float =0.5
):
    """
    Plots one heatmap for each method stored in success_matrixes 
    (keys are method names, values are success_matrix).
    All subplots share the same color scale (Success Ratio in [0,1])
    and a single colorbar.
    """

    font_size = 30
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size
    # Adjust the figure size as desired:
    num_methods = len(success_matrixes)
    fig, axs = plt.subplots(
        num_methods, 1,
        figsize=(12, 6*num_methods),  # scale width by #methods
        dpi=300,
        squeeze=False  # so axs is always 2D: shape (1, num_methods)
    )

    # For consistent color scale across subplots:
    vmin, vmax = 0, 1

    # We'll keep track of the mappable (the last image) to create the colorbar
    im = None

   # Plot each method's heatmap in its own row
    methods = success_matrixes
    for i, method in enumerate(methods):
        ax = axs[i, 0]
        success_matrix = success_matrixes[method]

        # Show the heatmap
        im = ax.imshow(
            success_matrix,
            cmap='Purples',
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
            extent=(0, max_corr, 0, len(d_trials))
        )

       # Major ticks at [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        major_ticks = np.arange(0, max_corr+0.01, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f"{val:.1f}" for val in major_ticks])

        # Minor ticks every 0.02 in [0..0.5], excluding major ticks
        all_ticks = np.arange(0, max_corr+0.02, 0.02)
        minor_ticks = [t for t in all_ticks if t not in major_ticks]
        ax.set_xticks(minor_ticks, minor=True)

        # y-axis: one tick per row in success_matrix (or use d/(2n), etc.)
        ax.set_yticks(range(len(d_trials)))
        ax.set_yticklabels([f"{d/(2*n):.0f}" for d in d_trials])

        # Label only the leftmost subplot's y-axis to avoid duplication:
        if i == num_methods - 1:
            # Label every subplot's x-axis:
            ax.set_xlabel(r"Corruption Level", fontsize=font_size)
            
        if i == num_methods//2:
            ax.set_ylabel(r"Measurement Ratio $m / (2n)$", fontsize=font_size)

        # Give each subplot a title corresponding to its method:
        ax.set_title(method, fontsize=font_size)
        
    # Create one colorbar for all subplots, using the last im
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.set_ylabel("Success Ratio", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    #plt.tight_layout()
    save_path = os.path.join(base_dir, f"{problem}_transition_plot.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"Figure saved to: {save_path}")
    plt.show()
    
def gen_random_point_in_neighborhood(X_true, radius, r, r_true):
    n, dim = X_true.shape
    
    padding = np.zeros((n, r - r_true))
    X_padded = np.hstack((X_true, padding))
    
    directions = np.random.normal(0, 1, (n, r))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    
    distances = np.random.uniform(0, 1, n) ** (1/r) * radius
    
    random_points = X_padded + directions * distances[:, np.newaxis]
    
    
    return random_points



def generate_matrix_with_condition(n, r, condition_number):
    
    A = np.random.randn(n, r)
    
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    s = np.linspace(1/condition_number, 1, min(n, r))
    
    S = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(S, s)
    
    A_prime = U @ S @ Vt
    
    return A_prime



def create_rip_transform(n, d, identity=False):
    if identity: 

        transformation_matrix = np.eye(n*n)
        adjoint_transformation_matrix = transformation_matrix.T

        def transform(matrix):
            # Flatten the matrix directly
            return np.dot(transformation_matrix, matrix.reshape(-1))
 
        def adjoint_transform(vector):
            # Reshape the vector back to the original matrix shape
            v = np.dot(adjoint_transformation_matrix, vector)
            return v.reshape(n, n)


    else:
        flattening_factor = n **2
    
        transformation_matrix_diag = np.random.normal(0, np.sqrt(1/d), (d, flattening_factor))
        transformation_matrix_offdiag = np.random.normal(0, np.sqrt(1/2*d), (d, flattening_factor))
        
        mask_diag = np.eye(transformation_matrix_diag.shape[0], transformation_matrix_diag.shape[1])
        mask_offdiag = 1 - mask_diag
        transformation_matrix = transformation_matrix_diag*mask_diag + transformation_matrix_offdiag*mask_offdiag
        adjoint_transformation_matrix = transformation_matrix.T
        
    
        def transform(matrix):
            matrix_flattened = matrix.reshape(-1)
            return np.dot(transformation_matrix, matrix_flattened) 
    
    
        def adjoint_transform(vector):
            matrix_reconstructed = np.dot(adjoint_transformation_matrix, vector)
            return matrix_reconstructed.reshape(n, n)

    return transform, adjoint_transform




def truncate_svd(matrix, k):
    """
    Perform truncated SVD on a matrix and return the truncated components.

    Parameters:
    - matrix: A 2D NumPy array.
    - k: The number of singular values/vectors to keep.

    Returns:
    - U_k: A matrix with the top k left singular vectors.
    - S_k: The top k singular values.
    - VT_k: A matrix with the top k right singular vectors, transposed.
    """
    U, S, VT = svd(matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]
    S_k_diag = np.diag(S_k)
    
    return np.dot(U_k, np.dot(S_k_diag, VT_k))



def matrix_recovery(
    X0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true,
    cond_number, method, base_dir, trial, gamma_init=0.00001, damping_init=0.1, q=0.8 ):
    """
    If restarted=False, run the original (non-restarted) method you provided.
    If restarted=True, run a 'Restarted Levenberg-Marquardt Subgradient Method'.
    """
    ###########################################################################
    # Common definitions
    ###########################################################################
    def c(x):
        return x @ x.T

    def h(M):
        if loss_ord == 0.5:
            return np.linalg.norm(A(M)-y_true)
        return (np.linalg.norm(A(M)-y_true, ord=loss_ord)**loss_ord)/loss_ord

    def subdifferential_h(M):
        """
        Returns subgradient (or gradient) in vectorized form (dimension n^2).
        """
        residual = A(M) - y_true

        # We assume A is a linear operator, e.g. A(M) = transformation_matrix * vec(M).
        # Using closures/tricks to get the matrix from A:
        transformation_matrix = A.__closure__[0].cell_contents  
        adjoint_A = transformation_matrix.T  

        subdiff = np.sign(residual) 
        
        if loss_ord == 1:
            v = np.dot(adjoint_A, subdiff)
        elif loss_ord == 2:
            v = np.dot(adjoint_A, residual)
        elif loss_ord == 0.5:
            # subgradient scaled by 1/h(M)
            v = np.dot(adjoint_A, residual) / (h(M))
        else:
            # You can extend as needed
            raise ValueError("Unsupported loss_ord")

        return v

    def operator(X, g):
        """
        Provided in your snippet. We keep it as is.
        X is n-by-r, g is n*r (vectorized).
        """
        g_mat = g.reshape(X.shape)
        return (2*g_mat @ X.T @ X + 2*X @ g_mat.T @ X).reshape(-1)


    #######################################################################
    # (A) ORIGINAL NON-RESTARTED VERSION
    #######################################################################
    X = X0.copy()
    losses = []

    n, r = X.shape
    for t in range(n_iter):

        if False:
            print("symmetric")
            print(f'Iteration number :  {t}')
            print(f'Method           :  {method}')
            print(f'Cond. number     :  {cond_number}')
            print(f"r*, r            :  {(r_true, r)}")
            val_cX = h(c(X))
            print(f'h(c(X)) = {"(DIVERGE)" if (np.isnan(val_cX) or val_cX == np.inf ) else val_cX}')
            print('---------------------')

        val_cX = h(c(X))
        if np.isnan(val_cX) or val_cX == np.inf or \
           np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star) > 1e4:
            # Indicate divergence in the losses
            losses += [1]*(n_iter - len(losses))
            break
        elif np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star) <= 1e-14:
            # Indicate we've basically converged
            losses += [1e-15]*(n_iter - len(losses))
            break
        else:
            losses.append(np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star))

        # Subgradient, etc.
        v = subdifferential_h(c(X))
        v_mat = v.reshape(X.shape[0], X.shape[0])
        g = ((v_mat + v_mat.T) @ X).reshape(-1)
        
        constant_stepsize = 1e-8 if X.shape[0] == 100 else 1e-6

        # Method logic (scaled subgrad, Gauss-Newton, etc.)
        if method == 'Scaled gradient' or method == 'Scaled subgradient' \
           or method == 'Precond. gradient'  \
           or match_method_pattern(method, prefix='Scaled gradient')[0] \
           or match_method_pattern(method, prefix='OPSA')[0]:
            
            if method in [ 'Scaled gradient', 'Scaled subgradient' ]:
                damping = 0
            elif method == 'Precond. gradient':
                damping = np.sqrt( h(c(X))) /4000 if loss_ord == 2 else h(c(X))/4000
            elif match_method_pattern(method, prefix='Scaled gradient')[0]:
                damping = convert_to_number(match_method_pattern(method, prefix='Scaled gradient')[1])
            elif match_method_pattern(method, prefix='OPSA')[0]:
                damping = convert_to_number(match_method_pattern(method, prefix='OPSA')[1])
            
            precondionner_inv = np.linalg.inv(X.T@X + damping*np.eye(r))
            
            G = g.reshape(*X.shape) \
                if not match_method_pattern(method, prefix='OPSA')[0] \
                else (g.reshape(*X.shape) + damping*X)

            preconditionned_G = G @ precondionner_inv
            aux = G @ sqrtm(precondionner_inv)
            
            if method in ['Scaled subgradient'] or match_method_pattern(method, prefix='OPSA')[0]:
                gamma = (h(c(X)) - 0) / np.sum(aux*aux)
            else:
                gamma = constant_stepsize

        elif method in ['Gauss-Newton', 'Levenberg–Marquardt (ours)']:

            if method == 'Levenberg–Marquardt (ours)':
                damping = np.sqrt( h(c(X))) /4000 if loss_ord == 2 else h(c(X))/100000
                damping = damping_init*q**t #decaying parameter
            else:
                damping = 0
            preconditionned_g = compute_preconditionner_applied_to_g_bm(X, g, damping)

            preconditionned_G = preconditionned_g.reshape(n, r)

            if method == 'Levenberg–Marquardt (ours)':
                gamma = (h(c(X)) - 0) / np.dot(v, v)
            elif method == 'Gauss-Newton':
                gamma = (h(c(X)) - 0) / np.dot(operator(X, g), preconditionned_g)

        elif method in ['Subgradient descent', 'Gradient descent', 'Polyak Subgradient']:
            preconditionned_G = g.reshape(n, r)
            gamma = (h(c(X)) - 0) / np.sum(g*g)

        else:
            raise NotImplementedError

        # Gradient-like update
        gamma = gamma_init*q**t #geometrically decaying stepsize
        X = X - gamma*preconditionned_G

    # Save results
    file_name = f'experiments/expbm_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    return losses




def compute_preconditionner_applied_to_g_bm(X,g, damping, max_iter=100, epsilon = 10**-13):
    """
    conjugate gradient method
    """
    def operator(X, g):
        #X is n by r, g is nr 
        g_mat = g.reshape(X.shape)
        return (2*g_mat@X.T@X + 2*X@g_mat.T@X).reshape(-1)
    
    n = X.shape[0]*X.shape[1]

    x = np.zeros_like(g)


    r = g - (operator(X, x) + damping*x)
    p = r.copy()
    rs_old = np.dot(r, r)

    info = {'iterations': 0, 'residual_norm': np.sqrt(rs_old)}

    for i in range(max_iter):
        Ap = operator(X, p) + damping*p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        rs_new = np.dot(r, r)
        info['iterations'] = i + 1
        info['residual_norm'] = np.sqrt(rs_new)
        if np.sqrt(rs_new) <= 10**-13:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x







def compute_preconditionner_applied_to_g_ass(X,Y, gx, gy, damping, max_iter=100, epsilon = 10**-13):
    """
    conjugate gradient method
    """

    def operator(X, Y, gx, gy):
        #Computes nabla c(X,Y).T * nabla c(X,Y) \cdot vect(gx, gy) 
        g_mat_x = gx.reshape(X.shape)
        g_mat_y = gy.reshape(Y.shape)
        
        return np.hstack( ((X@g_mat_y.T@Y + g_mat_x @ Y.T @ Y).reshape(-1), (Y@g_mat_x.T@X + g_mat_y @ X.T @ X).reshape(-1)))
        
    

    x = np.zeros_like(gx)
    y = np.zeros_like(gy)
    
    inp = np.hstack((x,y))
    
    g = np.hstack((gx, gy))
    


    r = g - (operator(X, Y, x, y) + damping*inp)
    p = r.copy()
    rs_old = np.dot(r, r)

    info = {'iterations': 0, 'residual_norm': np.sqrt(rs_old)}

    for i in range(max_iter):
        Ap = operator(X, Y, p[:len(x)], p[len(x): len(x) + len(y)]) + damping*p
        alpha = rs_old / np.dot(p, Ap)
        inp += alpha * p
        x = inp[:len(x)]
        y = inp[len(x): len(x) + len(y)]
        r -= alpha * Ap

        rs_new = np.dot(r, r)
        info['iterations'] = i + 1
        info['residual_norm'] = np.sqrt(rs_new)
        if np.sqrt(rs_new) <= 10**-13:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x,y



def matrix_recovery_assymetric(X0, Y0,Xstar,Ystar, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial, 
                               gamma_init=0.001, damping_init=0.0001, q=0.8):
    
    def c(X,Y):
        return X @ Y.T 

    def h(M):
        return (np.linalg.norm(A(M)-y_true,ord=loss_ord)**loss_ord)/loss_ord
        
    def subdifferential_h(M):
        
        residual = A(M) - y_true
        
        transformation_matrix = A.__closure__[0].cell_contents  # Extract the transformation matrix used in A
        adjoint_A = transformation_matrix.T  # Adjoint (transpose) of the transformation matrix
        

    
        subdiff = np.sign(residual) 
        
        if loss_ord == 1:
            v = np.dot(adjoint_A, subdiff)
        elif loss_ord == 2:
            v = np.dot(adjoint_A, residual)

        


        return v # n^2 dimension
    
    def operator(X, Y, gx, gy):
        #Computes nabla c(X,Y).T * nabla c(X,Y) \cdot vect(gx, gy) 
        g_mat_x = gx.reshape(X.shape)
        g_mat_y = gy.reshape(Y.shape)
        
        return np.hstack( ((X@g_mat_y.T@Y + g_mat_x @ Y.T @ Y).reshape(-1), (Y@g_mat_x.T@X + g_mat_y @ X.T @ X).reshape(-1)))
        
    
    
    
    
    X = X0.copy()
    Y = Y0.copy()
    losses = []



    n,r = X.shape
    for t in range(n_iter):
        
        if t%20 ==0:
            
            print("assymetric")
            print(f'Iteration number :  {t}')
            print(f'Method           :  {method}')
            print(f'Cond. number     :  {cond_number}')
            print(f"r*, r            :  {(r_true, r)}")
            print(f'h(c(X,Y)) = {"(DIVERGE)" if(np.isnan(h(c(X,Y)) ) or  (h(c(X,Y)) == np.inf) ) else  h(c(X,Y))}')
            print('---------------------')
     
        if np.isnan(h(c(X,Y)) ) or h(c(X,Y)) == np.inf or h(c(X,Y)) > 10**5:
            losses =  losses + [ 1 ] * (n_iter - len(losses)) #indicate divergence
            break
        elif np.linalg.norm(c(X,Y) - M_star)/np.linalg.norm(M_star) <= 10**-14:
            losses =  losses + [ 10**-15 ] * (n_iter - len(losses)) 
            break
        else:
            losses.append(np.linalg.norm(c(X,Y) - M_star)/np.linalg.norm(M_star))
        
        v = subdifferential_h(c(X,Y))
        
        g_x = (v.reshape((X.shape[0], Y.shape[0])) @ Y).reshape(-1)
        g_y = (v.reshape((X.shape[0], Y.shape[0])).T @ X).reshape(-1)
        
       
        
        constant_stepsize = 0.0000001 #if sensing else 0.1
        if method in [ 'Gauss-Newton', 'Levenberg–Marquardt (ours)']:
            damping = (np.sqrt( h(c(X,Y))) /4000 if loss_ord == 2 else h(c(X,Y))/100000) if method == 'Levenberg–Marquardt (ours)' else 0
            damping = damping_init*q**t if method == 'Levenberg–Marquardt (ours)' else 0
   
            preconditionned_g_x, preconditionned_g_y = compute_preconditionner_applied_to_g_ass(X, Y, g_x, g_y, damping)  
            preconditionned_G_x = preconditionned_g_x.reshape(*X.shape)
            preconditionned_G_y = preconditionned_g_y.reshape(*Y.shape)
            if method == 'Gauss-Newton':
                cctg =  operator(X, Y, preconditionned_g_x,  preconditionned_g_y)
                dir1 = cctg[:X.shape[0]*r]
                dir2 = cctg[X.shape[0]*r:]
                
                gamma = (h(c(X,Y)) - 0) / ( np.dot(dir1,preconditionned_g_x)  + np.dot(dir2, preconditionned_g_y))
                
            elif method == 'Levenberg–Marquardt (ours)':
                gamma = (h(c(X,Y)) - 0) / ( np.dot(v,v) )
            else:
                gamma = constant_stepsize
          
        
        elif method in ['Scaled gradient' ,'Scaled subgradient', 'Precond. gradient'] or match_method_pattern(method, prefix='Scaled gradient')[0] or match_method_pattern(method, prefix='OPSA')[0]: #PAPER: Low-Rank Matrix Recovery with Scaled subgradient Methods
            
           
           if method == 'Scaled gradient' or method=='Scaled subgradient':
               damping = 0
           elif method == 'Precond. gradient':
               damping = np.sqrt( h(c(X,Y))) /4000 if loss_ord == 2 else h(c(X,Y))/4000
           elif match_method_pattern(method, prefix='Scaled gradient')[0]:
               damping = convert_to_number( match_method_pattern(method, prefix='Scaled gradient')[1])
           elif match_method_pattern(method, prefix='OPSA')[0]:
               damping = convert_to_number( match_method_pattern(method, prefix='OPSA')[1])
        
           preconditionner_x = np.linalg.inv(Y.T@Y + damping* np.eye(X.shape[1]))
           preconditionner_y = np.linalg.inv(X.T@X + damping* np.eye(Y.shape[1]))
           G_x = g_x.reshape(*X.shape) if not match_method_pattern(method, prefix='OPSA')[0] else (g_x.reshape(*X.shape) + damping*X) 
           G_y = g_y.reshape(*Y.shape) if not match_method_pattern(method, prefix='OPSA')[0] else (g_y.reshape(*Y.shape) + damping*Y)
           preconditionned_G_x = G_x @ preconditionner_x 
           preconditionned_G_y = G_y @ preconditionner_y
           preconditionned_g_x = preconditionned_G_x.reshape(-1)
           preconditionned_g_y = preconditionned_G_y.reshape(-1)
           
           aux_x = G_x @ sqrtm(preconditionner_x) 
           aux_y = G_y @ sqrtm(preconditionner_y) 
           reg_diff = damping*0.5*( np.linalg.norm(X)**2 + np.linalg.norm(Y)**2 - np.linalg.norm(Xstar)**2 -  np.linalg.norm(Ystar)**2 if match_method_pattern(method, prefix='OPSA')[0] else 0)
           
           gamma = ((h(c(X,Y)) - 0)   / (np.sum(np.multiply(aux_x,aux_x)) + np.sum(np.multiply(aux_y, aux_y)))) if method in ['Scaled subgradient'] or match_method_pattern(method, prefix='OPSA')[0] else constant_stepsize
        
        elif method == 'Subgradient descent' or method =='Gradient descent': 
            preconditionned_G_x = g_x.reshape(*X.shape)
            preconditionned_G_y = g_y.reshape(*Y.shape)
            gamma = (h(c(X,Y)) - 0) / ( np.sum(np.multiply( g_x, g_x)) + np.sum(np.multiply( g_y, g_y)))
        
        else:

            raise NotImplementedError
        gamma = gamma_init*q**t
        X = X - gamma*preconditionned_G_x
        Y = Y - gamma*preconditionned_G_y
        
        if match_method_pattern(method, prefix='OPSA')[0]:
            X,Y = rebalance(X,Y)
        
        
    file_name = f'experiments/expasymmetric_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
    
    
    return losses

def rebalance(X, Y):
    # QR decompositions of X and Y
    QL, WL = np.linalg.qr(X)
    QR, WR = np.linalg.qr(Y)
    
    # SVD of the product WL @ WR.T
    U, s, Vt = np.linalg.svd(WL @ WR.T)

    # Create a rectangular diagonal matrix for singular values
    Sigma = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(Sigma, s)

    # Compute the rebalanced matrices
    L_OPSAd = QL @ U @ np.sqrt(Sigma)
    R_OPSAd = QR @ Vt.T @ np.sqrt(Sigma)
    
    return L_OPSAd, R_OPSAd

import torch
def retrieve_tensors(concatenated_tensor, original_shapes):

   
    lengths = [torch.prod(torch.tensor(shape)).item() for shape in original_shapes]

    split_tensors = torch.split(concatenated_tensor, lengths)

    return [split_tensors[i].reshape(original_shapes[i]) for i in range(len(original_shapes))]


def random_perturbation(dimensions, delta):
    """
    Generates a random perturbation tensor with specified dimensions and a Frobenius norm of at least delta.

    Args:
    dimensions (tuple of ints): The shape of the tensor to generate.
    delta (float): The minimum Frobenius norm of the generated tensor.

    Returns:
    torch.Tensor: A tensor of the specified dimensions with a Frobenius norm of at least delta.
    """
    # Generate a random tensor with the specified dimensions
    perturbation = torch.randn(dimensions)

    # Calculate its current Frobenius norm
    current_norm = torch.norm(perturbation, p='fro')

    # Scale the tensor to have a Frobenius norm of exactly delta
    if current_norm > 0:
        scale_factor = delta / current_norm
        perturbation *= scale_factor
    
    # If current norm is exactly zero (rare), regenerate the tensor (recursive call)
    if current_norm == 0:
        return random_perturbation(dimensions, delta)

    return perturbation

def fill_tensor_elementwise(source, target):
    """Fill target tensor element-wise from source tensor."""
    if len(source.shape) == 1:
        for i in range(source.shape[0]):
            target[i] = source[i]
    elif len(source.shape) == 2:
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                target[i, j] = source[i, j]
    elif len(source.shape) == 3:
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                for k in range(source.shape[2]):
                    target[i, j, k] = source[i, j, k]
                    
def thre(inputs, threshold, device):
    return torch.sign(inputs) * torch.max( torch.abs(inputs) - threshold, torch.zeros(inputs.shape).to(device))


import re

def convert_to_number(latex_str):
    """
    Converts a string in LaTeX-like format (e.g., '10^{-3}') to a numerical value.
    
    Args:
        latex_str (str): A string representing a number in LaTeX-like format.
    
    Returns:
        float: The numerical value of the input string.
    """
    # Replace LaTeX exponent format with Python exponentiation
    python_expr = re.sub(r'\^{(-?\d+)}', r'**\1', latex_str)
    # Evaluate and return the numerical result
    return eval(python_expr)

def match_method_pattern(method, prefix='Scaled gradient'):
    """
    Checks if the method string matches the pattern '<prefix>($<string>$)'.
    
    Args:
        method (str): The method string to check.
        prefix (str): The prefix to match at the start of the method string. Default is 'Scaled gradient'.
    
    Returns:
        tuple:
            - bool: True if the pattern matches, False otherwise.
            - str or None: The inner string if matched; otherwise, None.
    """
    # Escape the prefix to handle any special regex characters
    escaped_prefix = re.escape(prefix)
    
    # Define the regex pattern dynamically based on the prefix
    pattern = rf'^{escaped_prefix}\(\$\\lambda=(.*?)\$\)$'
    
    # Attempt to match the pattern
    match = re.match(pattern, method)
    
    if match:
        # Extract the inner string using capturing group
        inner_string = match.group(1)
        return True, inner_string
    else:
        return False, None

