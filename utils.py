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
    
            radius_x = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
            radius_y = init_radius_ratio*np.linalg.norm(Y_true, ord='fro')


            X_padded = np.zeros((n, r))
            Y_padded = np.zeros((n, r))
            X_padded[:, :r_true] = X_true
            Y_padded[:, :r_true] = Y_true
            
            X0 = gen_random_point_in_neighborhood(X_true, radius_x, r, r_true)
            Y0 = gen_random_point_in_neighborhood(Y_true, radius_y, r, r_true)
            

        
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
    mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed and desired

    # Adjust matplotlib parameters for publication quality
    mpl.rcParams['figure.figsize'] = (8, 6)  # Increased size for better visibility
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 6

    # Define a color palette commonly used in optimization papers (Tableau 10)
    tableau_colors = [
        '#4d4d4d', #BASELINE
        '#0072b2',  #OUR 
        '#e69f00',   #COMPETITOR 1
        '#009e73',   #COMPTEITOR 2
        '#d55e00'] #FAILING METHOD
    methods = list(losses.keys())
    for m in methods:
        for m2 in methods:
            assert losses[m].keys() == losses[m2].keys(), "All methods must have the same keys."

    # Ensure enough colors for each method
    assert len(tableau_colors) >= len(methods), "Not enough colors for the number of methods."

    markers = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']  # Extended markers list
    linestyles = ['-', ':']  # Different linestyles

    fig, ax = plt.subplots()

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
            convergence_threshold = 1e-13   # Slightly above machine epsilon to account for numerical errors
            converged_indices = np.where(errs <= convergence_threshold)[0]

            if converged_indices.size > 0:
                last_index = converged_indices[0] + 1  # Include the converged point
            else:
                last_index = len(errs)

            # Slice the errors up to the convergence point
            errs = errs[:last_index]
            std = std[:last_index]

            #
            num_dots_adapted = int(num_dots*last_index)
     
            
            start = int( idx_m * ( num_dots_adapted )/(len(methods)))
            
            if k[1] > 1:
                start += num_dots_adapted/4
                if k[0] > r_true:
                    start += num_dots_adapted/4
            start = int(start)
           
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
                linestyle = linestyles[1 if k[0] > r_true else 0 ]
                linestyle_styles[over_exact_label] = linestyle  # Store linestyle for the parameterization
            else:
                linestyle = linestyle_styles[over_exact_label]
            
            # Determine marker based on ill-conditioning (kappa value)
            kappa_label = "Ill-conditioned" if k[1] > 1 else "Well-conditioned"
            if kappa_label not in marker_styles:
                marker = markers[ 1 if k[1] > 1 else 0]
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
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel(r'Relative Distance to $z^\ast$', fontsize=14)
    ax.set_yscale('log')

    # Adding grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjusting tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.5)

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

    # Place legends
    # Methods legend at upper right
    legend1 = ax.legend(
        method_handles,
        method_labels,
        title='Methods',
        loc='upper right',
        bbox_to_anchor=(0.693, 0.976),
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )

    # Combined legend for markers and linestyles at lower left
    combined_handles = marker_handles + linestyle_handles
    combined_labels = marker_labels + linestyle_labels
    legend2 = ax.legend(
        combined_handles,
        combined_labels,
        title='Setting',
        loc='upper right',
       bbox_to_anchor=(1, 1),  # Adjust the y-coordinate as needed
        frameon=True,
        facecolor='white',
        edgecolor='black'
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



    #X X^T flattened jacobian
    def jacobian_c(X):
        n, r = X.shape
        jac = np.zeros((n*n, n*r))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(r):
                        # Derivative of c_ij with respect to x_kl
                        if i != j:
                            if i ==k:
                                jac[i*n + j, k*r + l] = X[j, l]
                            if j ==k:
                                jac[i*n + j, k*r + l] = X[i, l]
                                
                                
                        if i == j:
                            if i == k:
                                jac[i*n + j, k*r + l] = 2*X[i, l]
                                
        
        return jac #n^2 X nr matrix
    
    def update_jacobian_c(jac, X):
        n, r = X.shape
        
        # Update for i != j
        for i in range(n):
            for j in range(n):
                if i != j:
                    for l in range(r):
                        # Update for i == k
                        jac[i*n + j, i*r + l] = X[j, l]
                        # Update for j == k
                        jac[i*n + j, j*r + l] = X[i, l]
    
        # Update for i == j
        for i in range(n):
            for l in range(r):
                # Update for i == k
                jac[i*n + i, i*r + l] = 2*X[i, l]

        
        
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
    losses = []
    jacob_c = jacobian_c(X)

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

        
        if np.isnan(h(c(X)) ) or h(c(X)) == np.inf:
            losses.append(10**10)
        else:
            losses.append(np.linalg.norm(c(X) - M_star)/np.linalg.norm(c(X0) - M_star))
        
        update_jacobian_c(jacob_c, X)
        
        
        v = subdifferential_h(c(X))
        
        g = jacob_c.T @ v
        
        
        
        
        constant_stepsize = 0.000001 #if sensing else 0.1 #if sensing else 0.1
            
        if method == 'scaled_gd' or method == 'scaled_subgd' or method == 'pred_gd'  or method == 'scaled_gd($\lambda$)' or method == 'rebalance': #PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
        
           
            if method == 'scaled_gd' or method=='scaled_subgd':
                dampling = 0
            elif method == 'pred_gd':
                dampling = np.linalg.norm(c(X) - M_star, ord='fro')
            elif method == 'scaled_gd($\lambda$)' or method == 'rebalance':
                dampling = 10**-9
        
            precondionner_inv =   np.linalg.inv( X.T@X + dampling* np.eye(r) )  #(np.inv()) for scaled
            
            
            G = g.reshape(*X.shape) if method != 'rebalance' else (g.reshape(*X.shape) + dampling*X) 
            preconditionned_G = G @ precondionner_inv
            aux = G @ sqrtm(precondionner_inv)
            gamma = (h(c(X)) - 0) / (np.sum(np.multiply(aux,aux)))  if method in ['scaled_subgd', 'rebalance'] else constant_stepsize
            #constant 
            #gamma = 0.000001
            
               
        elif method=='gnp' or method=='gn':
            
            try:
                dampling = np.linalg.norm(c(X) - M_star, ord='fro')
                preconditionned_g, _,_,_ = np.linalg.lstsq(jacob_c.T @ jacob_c + dampling*np.eye(jacob_c.shape[1],jacob_c.shape[1]), g, rcond=-1)
            except:
                preconditionned_g = g #No precondionning 
                
            preconditionned_G = preconditionned_g.reshape(n,r)
            gamma = 1.5*(h(c(X)) - 0) / np.dot(v,v) if  method == 'gnp' else 2*constant_stepsize
        elif method == 'vanilla_subgd' or method == 'vanilla_gd':
            preconditionned_G  = g.reshape(n,r)
            gamma = (h(c(X)) - 0) / np.sum(np.multiply(g,g)) if method == 'vanilla_subgd' else constant_stepsize
            

        else:
            raise NotImplementedError
        

        X = X - gamma*preconditionned_G
        if method == 'rebalance':
            X,_ = rebalance(X,X)
    
    file_name = f'experiments/expbm_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
    
    
    return losses



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

        
        if np.isnan(h(c(X,Y)) ) or h(c(X,Y)) == np.inf:
            losses.append(10**10)
        else:
            losses.append(np.linalg.norm(c(X,Y) - M_star)/(np.linalg.norm(c(X0,Y0) - M_star)))
        
        jac_x, jac_y = jacobian_c(X, Y)
        
        
        v = subdifferential_h(c(X,Y))
        
        g_x = jac_x.T @ v
        g_y = jac_y.T @ v
        
       
        
        constant_stepsize = 0.000001 #if sensing else 0.1
        if method=='gnp' or method == 'gn':
            dampling = np.linalg.norm(c(X,Y) - M_star, ord='fro')
            
            try:
                
                preconditionned_g_x, _,_,_ = np.linalg.lstsq(jac_x.T @ jac_x + dampling*np.eye(jac_x.shape[1],jac_x.shape[1]), g_x, rcond=-1)
                preconditionned_g_y, _,_,_ = np.linalg.lstsq(jac_y.T @ jac_y + dampling*np.eye(jac_y.shape[1],jac_y.shape[1]), g_y, rcond=-1)
            except:
                preconditionned_g_x = g_x #No precondionning 
                preconditionned_g_y = g_y
                
            preconditionned_G_x = preconditionned_g_x.reshape(*X.shape)
            preconditionned_G_y = preconditionned_g_y.reshape(*Y.shape)
            gamma = 1.5*(h(c(X,Y)) - 0) / ( np.dot(v,v) )  if method == 'gnp' else constant_stepsize
          
        
        elif method in ['scaled_gd' ,'scaled_subgd', 'pred_gd', 'scaled_gd($\lambda$)', 'rebalance']: #PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
            
           
           if method == 'scaled_gd' or method=='scaled_subgd':
               dampling = 0
           elif method == 'pred_gd':
               dampling = np.linalg.norm(c(X,Y) - M_star, ord='fro')
           elif method == 'scaled_gd($\lambda$)' or method == 'rebalance':
               dampling = 10**-9
        
        
           preconditionner_x = np.linalg.inv(Y.T@Y + dampling* np.eye(X.shape[1]))
           preconditionner_y = np.linalg.inv(X.T@X + dampling* np.eye(Y.shape[1]))
           G_x = g_x.reshape(*X.shape) if method != 'rebalance' else (g_x.reshape(*X.shape) + dampling*X) 
           G_y = g_y.reshape(*Y.shape) if method != 'rebalance' else (g_y.reshape(*Y.shape) + dampling*Y)
           preconditionned_G_x = G_x @ preconditionner_x 
           preconditionned_G_y = G_y @ preconditionner_y
           preconditionned_g_x = preconditionned_G_x.reshape(-1)
           preconditionned_g_y = preconditionned_G_y.reshape(-1)
           
           aux_x = G_x @ sqrtm(preconditionner_x) 
           aux_y = G_y @ sqrtm(preconditionner_y) 
           reg_diff = dampling*0.5*( np.linalg.norm(X)**2 + np.linalg.norm(Y)**2 - np.linalg.norm(Xstar)**2 -  np.linalg.norm(Ystar)**2 if method == 'rebalance' else 0)
           
           gamma = ((h(c(X,Y)) - 0)   / (np.sum(np.multiply(aux_x,aux_x)) + np.sum(np.multiply(aux_y, aux_y)))) if method in ['scaled_subgd',
                                                                                                                          'rebalance'] else constant_stepsize
        
        elif method == 'vanilla_subgd' or method =='vanilla_gd': 
            preconditionned_G_x = g_x.reshape(*X.shape)
            preconditionned_G_y = g_y.reshape(*Y.shape)
            gamma = (h(c(X,Y)) - 0) / ( np.sum(np.multiply( g_x, g_x)) + np.sum(np.multiply( g_y, g_y)))  if method == 'vanilla_subgd' else constant_stepsize
        
        else:

            raise NotImplementedError
        
        X = X - gamma*preconditionned_G_x
        Y = Y - gamma*preconditionned_G_y
        
        if method == 'rebalance':
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
    L_rebalanced = QL @ U @ np.sqrt(Sigma)
    R_rebalanced = QR @ Vt.T @ np.sqrt(Sigma)
    
    return L_rebalanced, R_rebalanced

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




