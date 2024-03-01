#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:53:34 2024

@author: aglabassi
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import sqrtm
import os

def plot_losses_with_styles(losses_scaled, losses_gnp, lambdaa_scaled, lambdaa_gnp, cond_numbers, ranks, r_true, loss_ord, base_dir, num_dots=20):
    # Define color palettes for 'scaled' (blue family) and 'gn' (red family)
 
    blue_palette = ['#0d47a1', '#1976d2', '#2196f3', '#64b5f6', '#bbdefb']
    # Shades of red for 'gn' (warm colors)
    red_palette = ['#b71c1c', '#e53935', '#ef5350', '#e57373', '#ffcdd2']


    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'x']  # Different markers for different condition numbers
    linestyles = ['-', '--', '-.', ':']  # Different linestyles for different ranks
    
    # Generate labels for plots
    labels_scaled = [f'scaled, cond_n={c}, r={r}' for c in cond_numbers for r in ranks]
    labels_gn = [f'gn, cond_n={c}, r={r}' for c in cond_numbers for r in ranks]
    labels = labels_scaled + labels_gn

    # Assign colors, markers, and linestyles
    blue_colors_cycle = itertools.cycle(blue_palette)
    red_colors_cycle = itertools.cycle(red_palette)
    plot_colors = [next(blue_colors_cycle) if i < len(labels_scaled) else next(red_colors_cycle) for i in range(len(labels))]

    markers_cycle = itertools.cycle(markers)
    plot_markers = {c: next(markers_cycle) for c in cond_numbers}
    linestyles_cycle = itertools.cycle(linestyles)
    plot_linestyles = {r: next(linestyles_cycle) for r in ranks}

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    lines = []  # To store line objects for the legend
    for i, label in enumerate(labels):
        color = plot_colors[i]
        cond_n, r = label.split(',')[1].strip(), label.split(',')[2].strip()
        marker = plot_markers[int(cond_n.split('=')[1])]
        linestyle = plot_linestyles[int(r.split('=')[1])]
        
        loss_data = losses_scaled[i] if i < len(losses_scaled) else losses_gnp[i - len(losses_scaled)]
        
        # Plot the line
        line, = plt.plot(loss_data, color=color, linestyle=linestyle)
        
        # Calculate indices for evenly spaced markers
        if len(loss_data) > 1:  # Ensure there's data to plot
            indices = np.round(np.linspace(0, len(loss_data) - 1, num_dots)).astype(int)
            # Plot markers at these indices
            plt.plot(indices, [loss_data[idx] for idx in indices], linestyle='None', marker=marker, color=color)
        
        # Add a dummy line to the list for creating a custom legend
        lines.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, label=label))
    
    plt.title(f'Loss function for Matrix Recovery, r_true = {r_true}, lambda_sc={lambdaa_scaled}, lambda_gnp={lambdaa_gnp}, loss=l{loss_ord}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    # Create a custom legend
    plt.legend(handles=lines)
    fig_path = os.path.join(base_dir, f'exp_l{loss_ord}.png')
    plt.savefig(fig_path, dpi=300)
    plt.show()

    




def gen_random_point_in_neighborhood(X_true, radius, r, r_true):
    n, dim = X_true.shape  # n is the number of points, dim is the dimensionality
    
    padding = np.zeros((n, r - r_true))
    X_padded = np.hstack((X_true, padding))
    
    # Step 1: Generate random directions
    directions = np.random.normal(0, 1, (n, r))
    # Normalize each vector to have unit length
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    
    # Step 2: Generate random distances from the center
    distances = np.random.uniform(0, 1, n) ** (1/r) * radius
    
    # Step 3: Scale the directions by the distances (need to reshape distances for broadcasting)
    random_points = X_padded + directions * distances[:, np.newaxis]
    
    
    #random_points = np.random.normal(0, 1/n, (n, r))*10**-1
    return random_points



def generate_matrix_with_condition(n, r, condition_number):
    # Generate a random matrix of dimension n x r
    A = np.random.randn(n, r)
    
    # Perform SVD on the matrix
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Adjust the singular values to achieve the desired condition number
    # Set the smallest singular value to the  inverse of condition number  and the biggest to 1
    s = np.linspace(1/condition_number, 1, min(n, r))
    
    # Construct the diagonal matrix of singular values
    S = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(S, s)
    
    # Reconstruct the matrix with the desired condition number
    A_prime = U @ S @ Vt
    
    return A_prime



def create_rip_transform(n, d):
    """
    Create a linear transformation that maps matrices of dimension n*n to vectors of dimension d.
    This uses a random Gaussian matrix, which is known to satisfy the RIP under certain conditions.
    """
    # Flattening factor to convert the matrix to a vector
    flattening_factor = n **2

    # Create a random Gaussian matrix of size d x (n*r)
    # The entries are drawn from N(0, 1/d) to ensure the RIP with high probability
    transformation_matrix_diag = np.random.normal(0, np.sqrt(1/d), (d, flattening_factor))
    transformation_matrix_offdiag = np.random.normal(0, np.sqrt(1/2*d), (d, flattening_factor))
    
    mask_diag = np.eye(transformation_matrix_diag.shape[0], transformation_matrix_diag.shape[1])
    mask_offdiag = 1 - mask_diag
    transformation_matrix = transformation_matrix_diag*mask_diag + transformation_matrix_offdiag*mask_offdiag
    adjoint_transformation_matrix = transformation_matrix.T
    

    # Define the linear transformation function
    def transform(matrix):
        # Flatten the matrix into a vector
        matrix_flattened = matrix.reshape(-1)
        # Apply the linear transformation
        return np.dot(transformation_matrix, matrix_flattened)


    def adjoint_transform(vector):
        # Apply the adjoint linear transformation, then reshape the result back into a matrix
        matrix_reconstructed = np.dot(adjoint_transformation_matrix, vector)
        return matrix_reconstructed.reshape(n, n)

    return transform, adjoint_transform






def matrix_recovery(X0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, lambdaa, method):
    
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


    X = X0
    losses = []
    jacob_c = jacobian_c(X)

    # Pre-compute A*, B*, C*
    #A_star, B_star, C_star = compute_iterate_decomposition(X_true_padded, U_star)
    n,r = X.shape
    for t in range(n_iter):
        
        if t%20 ==0:
            print(f'Iteration number :  {t}')
            print(f'Method           :  {method}')
            print(f'Cond. number     :  {cond_number}')
            print(f"r*, r            :  {(r_true, r)}")
            print(f'h(c(X)) = {"(DIVERGE)" if(np.isnan(h(c(X)) ) or h(c(X)) > 2*h(c(X0))) else  h(c(X))}')
            print('---------------------')

        
        if np.isnan(h(c(X)) ) or h(c(X)) > 2*h(c(X0)):
            losses.append(10**10)
        else:
            losses.append(h(c(X))/y_true.shape[0] )
        
        
        # Compute the Jacobian of c at x
        update_jacobian_c(jacob_c, X)
        
        
        # Compute v from the subdifferential of h at c(x)
        v = subdifferential_h(c(X))
        
        #subdifferential of h(c(x)) w.r.t x
        g = jacob_c.T @ v
        
        if method=='gnp':
            dampling = lambdaa if lambdaa != 'Liwei' else np.linalg.norm(c(X) - M_star, ord='fro')
            preconditionned_g, _,_,_ = np.linalg.lstsq(jacob_c.T @ jacob_c + dampling*np.eye(jacob_c.T.shape[0],jacob_c.T.shape[0]), g, rcond=None)
            aux = (jacob_c @ preconditionned_g)
            preconditionned_G = preconditionned_g.reshape(n,r)
            gamma = (h(c(X)) - 0) / np.dot(aux,aux) if loss_ord == 1 else 0.000001
              

            
        elif method=='scaled':
            try:
                dampling = lambdaa if lambdaa != 'Liwei' else np.linalg.norm(c(X) - M_star, ord='fro')
                preconditionned_G = A_adj((A(X@X.T) - y_true))@ X @ np.linalg.inv(X.T@X + dampling*np.eye(r,r)) if loss_ord==2 else A_adj(( np.sign(A(X@X.T) - y_true)) ) @ X @ np.linalg.inv(X.T@X + dampling*np.eye(r,r))
                preconditionned_g = preconditionned_G.reshape(-1)
                aux = A_adj(( np.sign(A(X@X.T) - y_true)) ) @ X @ sqrtm(np.linalg.inv(X.T@X + dampling*np.eye(r,r)))
                gamma = (h(c(X)) - 0) / np.sum(np.multiply(aux,aux)) if loss_ord == 1 else 0.000001
            except:
                ''
        else:
            raise NotImplementedError
        

        X = X - gamma*preconditionned_G
    

    return losses






