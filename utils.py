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



def trial_execution_matrix(trials, n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric=True, identity=False):
    
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
    
            for method in methods:
                if symmetric:
                    matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial)
                else:
                    matrix_recovery_assymetric(X0, Y0, X_true, Y_true, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial)
                    
    return 'we dont care'





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


    # Define a color palette commonly used in optimization papers (Tableau 10)
    tableau_colors_temp = [
        '#dc267f',  # baseline
        '#ffb000',  # COMPETITOR 1
        '#fe6100',  # COMPETITOR 2 
        '#648fff'   # COMPETITOR 3 
    ]
    our_color =   '#785ef0'  # OUR (Ultramarine 40)
    methods = list(losses.keys())
    
    tableau_colors = tableau_colors_temp[:len(methods)-1] + [our_color]
    for m in methods:
        for m2 in methods:
            assert losses[m].keys() == losses[m2].keys(), "All methods must have the same keys."

    # Ensure enough colors for each method
    assert len(tableau_colors) >= len(methods), "Not enough colors for the number of methods."

    markers = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']  # Extended markers list
    linestyles = ['-', ':']  # Different linestyles

    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=25)

    keys = list(losses[methods[0]].keys())

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
    method_colors = {}

    for idx_m, method in enumerate(methods):
        color = tableau_colors[idx_m % len(tableau_colors)]
        method_colors[method] = color  # Store color for the method


        for idx, k in enumerate(keys):
            r_index = rs.index(k[0])
            c_index = cs.index(k[1])
            errs = np.array(losses[method][k])
            std = np.array(stds[method][k])

            # Determine the index where errors have converged to machine epsilon
            convergence_threshold = 1e-12   # Slightly above machine epsilon to account for numerical errors
            converged_indices = np.where(errs <= convergence_threshold)[0]

            if converged_indices.size > 0:
                last_index = converged_indices[0] + 1  # Include the converged point
            else:
                print(errs)
                last_index = len(errs)

            # Slice the errors up to the convergence point
            errs = errs[:last_index]
            std = std[:last_index]

            num_dots_adapted = int(num_dots * last_index)

            start = 0

            indices = np.arange(start, last_index, num_dots_adapted)

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
    height = 0.61 if problem=='Burer-Monteiro' else 0.77
    print(problem)
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
    fig_path = os.path.join(base_dir, f'experiments/exp_l{loss_ord}_{problem}_{kappa}.pdf')
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')

    # Display the plot
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




def matrix_recovery(X0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial):
    
    def c(x):
        return x @ x.T 

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


    X = X0.copy()
    X_prev = X0.copy()
    losses = []

    n,r = X.shape
    for t in range(n_iter):
        
        if t%20 ==0:
            print("symmetric")
            print(f'Iteration number :  {t}')
            print(f'Method           :  {method}')
            print(f'Cond. number     :  {cond_number}')
            print(f"r*, r            :  {(r_true, r)}")
            print(f'h(c(X)) = {"(DIVERGE)" if(np.isnan(h(c(X)) ) or  (h(c(X)) == np.inf) ) else  h(c(X))}')
            print('---------------------')

        
        if np.isnan(h(c(X)) ) or h(c(X)) == np.inf or h(c(X)) > 10**5:
            losses =  losses + [ 1 ] * (n_iter - len(losses)) #indicate divergence
            break
        elif h(c(X)) <= 10**-9/2:
            losses =  losses + [ 10**-15 ] * (n_iter - len(losses)) 
            break
        else:
            losses.append(np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star))
        
        
        
        v = subdifferential_h(c(X))
        
        
        v_mat = v.reshape(X.shape[0], X.shape[0])
        
        g = (( v_mat + v_mat.T )@X).reshape(-1)

        
        
        
        
        constant_stepsize = 0.00000001 #if sensing else 0.1 #if sensing else 0.1
            
        if method == 'Scaled' or method == 'Scaled subgradient' or method == 'Precond. GD'  or match_method_pattern(method, prefix='Scaled')[0] or match_method_pattern(method, prefix='OPSA')[0]: #PAPER: Low-Rank Matrix Recovery with Scaled subgradient Methods
        
           
            if method in [ 'Scaled', 'Scaled subgradient' ]:
                damping = 0
            elif method == 'Precond. GD':
                damping = np.linalg.norm(c(X) - M_star, ord='fro')
            elif match_method_pattern(method, prefix='Scaled')[0]:
                damping = float(match_method_pattern(method, prefix='Scaled')[1])
            elif match_method_pattern(method, prefix='OPSA')[0]:
                damping = float(match_method_pattern(method, prefix='OPSA')[1])
        
            precondionner_inv =   np.linalg.inv( X.T@X + damping* np.eye(r) )  #(np.inv()) for Scaled
            
            
            G = g.reshape(*X.shape) if not  match_method_pattern(method, prefix='OPSA')[0] else (g.reshape(*X.shape) + damping*X) 
            preconditionned_G = G @ precondionner_inv
            aux = G @ sqrtm(precondionner_inv)
            gamma = (h(c(X)) - 0) / (np.sum(np.multiply(aux,aux)))  if method in ['Scaled subgradient'] or match_method_pattern(method, prefix='OPSA')[0]else constant_stepsize
            #constant 
            #gamma = 0.000001
            
               
        elif method in ['Gauss-Newton', 'Gauss-Newton, $\eta_k = \eta$', 'Levenberg–Marquard (ours)', 'Levenberg–Marquard (ours), $\eta_k = \eta$']:
            
            try:
                damping = np.linalg.norm(c(X) - M_star) if method in ['Levenberg–Marquard (ours)', 'Levenberg–Marquard (ours), $\eta_k = \eta$'] else 0
                preconditionned_g = compute_preconditionner_applied_to_v(X, g, damping)
            except:
                preconditionned_g = g #No precondionning 
                
            preconditionned_G = preconditionned_g.reshape(n,r)
            gamma = 1.5*(h(c(X)) - 0) / np.dot(v,v) if  method in ['Gauss-Newton', 'Levenberg–Marquard (ours)'] else 2*constant_stepsize
        elif method == 'Subgradient descent' or method == 'Gradient descent':
            preconditionned_G  = g.reshape(n,r)
            gamma = (h(c(X)) - 0) / np.sum(np.multiply(g,g)) if method == 'Subgradient descent' else constant_stepsize
            

        else:
            raise NotImplementedError
        
        X_prev = X
        X = X - gamma*preconditionned_G
    
    file_name = f'experiments/expbm_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
    
    
    return losses


def compute_preconditionner_applied_to_v(X,v, damping, max_iter=100, epsilon = 10**-13):
    """
    conjugate gradient method
    """
    n = X.shape[0]*X.shape[1]

    x = np.zeros_like(v)


    r = v - (compute_bm_jac_product(X, x) + damping*x)
    p = r.copy()
    rs_old = np.dot(r, r)

    info = {'iterations': 0, 'residual_norm': np.sqrt(rs_old)}

    for i in range(max_iter):
        Ap = compute_bm_jac_product(X, p) + damping*p
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



def compute_bm_jac_product(X, v):
    #X is n by r, v is nr 
    V_mat = v.reshape(X.shape)
    return (2*V_mat@X.T@X + 2*X@V_mat.T@X).reshape(-1)



def matrix_recovery_assymetric(X0, Y0,Xstar,Ystar, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial):
    
    def c(X,Y):
        return X @ Y.T 

    def h(M):
        return (np.linalg.norm(A(M)-y_true,ord=loss_ord)**loss_ord)/loss_ord



    #X Y^T flattened jacobian
    def jacobian_c(X,Y):
        
        m, r = X.shape
        n = Y.shape[0]
        jac_x = np.zeros((n*m, m*r))
        jac_y = np.zeros((n*m, n*r))
        
        for i in range(n):
            for j in range(n):
                for l in range(r):
                    jac_x[i*n + j, i*r + l] = Y[j, l]
                    
        
        for i in range(n):
            for j in range(n):
                    for l in range(r):
                        jac_y[i*n + j, j*r + l] = X[i, l]
                    
        
        #return np.kron(np.eye(X.shape[0]),Y), jac_y
        
        return  jac_x, jac_y
        
        
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

        
        if np.isnan(h(c(X,Y)) ) or h(c(X,Y)) == np.inf or h(c(X,Y)) > 10**3:
            losses.append(10**10)
        else:
            losses.append(np.linalg.norm(c(X,Y) - M_star)/(np.linalg.norm(M_star)))
        
        jac_x, jac_y = jacobian_c(X, Y)
        
        
        v = subdifferential_h(c(X,Y))
        
        g_x = jac_x.T @ v
        g_y = jac_y.T @ v
        
       
        
        constant_stepsize = 0.000001 #if sensing else 0.1
        if method in ['Gauss-Newton, $\eta_k = \eta$', 'Gauss-Newton', 'Levenberg–Marquard (ours), $\eta_k = \eta$', 'Levenberg–Marquard (ours)']:
            damping = np.linalg.norm(c(X,Y) - M_star)if method in [ 'Levenberg–Marquard (ours)', 'Levenberg–Marquard (ours), $\eta_k = \eta$'] else 0
            
            try:
                
                preconditionned_g_x, _,_,_ = np.linalg.lstsq(jac_x.T @ jac_x + damping*np.eye(jac_x.shape[1],jac_x.shape[1]), g_x, rcond=None)
                preconditionned_g_y, _,_,_ = np.linalg.lstsq(jac_y.T @ jac_y + damping*np.eye(jac_y.shape[1],jac_y.shape[1]), g_y, rcond=None)
            except:
                preconditionned_g_x = g_x #No precondionning 
                preconditionned_g_y = g_y
                
            preconditionned_G_x = preconditionned_g_x.reshape(*X.shape)
            preconditionned_G_y = preconditionned_g_y.reshape(*Y.shape)
            gamma = 1.5*(h(c(X,Y)) - 0) / ( np.dot(v,v) )  if method in ['Gauss-Newton', 'Levenberg–Marquard (ours)']  else constant_stepsize
          
        
        elif method in ['Scaled' ,'Scaled subgradient', 'Precond. GD'] or match_method_pattern(method, prefix='Scaled')[0] or match_method_pattern(method, prefix='OPSA')[0]: #PAPER: Low-Rank Matrix Recovery with Scaled subgradient Methods
            
           
           if method == 'Scaled' or method=='Scaled subgradient':
               damping = 0
           elif method == 'Precond. GD':
               damping = np.linalg.norm(c(X,Y) - M_star, ord='fro')
           elif match_method_pattern(method, prefix='Scaled')[0]:
               damping = float( match_method_pattern(method, prefix='Scaled')[1])
           elif match_method_pattern(method, prefix='OPSA')[0]:
               damping = float( match_method_pattern(method, prefix='OPSA')[1])
        
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
            gamma = (h(c(X,Y)) - 0) / ( np.sum(np.multiply( g_x, g_x)) + np.sum(np.multiply( g_y, g_y)))  if method == 'Subgradient descent' else constant_stepsize
        
        else:

            raise NotImplementedError
        
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

import re

def match_method_pattern(method, prefix='Scaled'):
    """
    Checks if the method string matches the pattern '<prefix>($<string>$)'.
    
    Args:
        method (str): The method string to check.
        prefix (str): The prefix to match at the start of the method string. Default is 'Scaled'.
    
    Returns:
        tuple:
            - bool: True if the pattern matches, False otherwise.
            - str or None: The inner string if matched; otherwise, None.
    """
    # Escape the prefix to handle any special regex characters
    escaped_prefix = re.escape(prefix)
    
    # Define the regex pattern dynamically based on the prefix
    pattern = rf'^{escaped_prefix}\(\$(.*?)\$\)$'
    
    # Attempt to match the pattern
    match = re.match(pattern, method)
    
    if match:
        # Extract the inner string using capturing group
        inner_string = match.group(1)
        return True, inner_string
    else:
        return False, None

