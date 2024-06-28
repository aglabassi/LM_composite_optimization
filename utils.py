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
from scipy.linalg import svd

def plot_losses_with_styles(losses_scaled, losses_gnp, lambdaa_scaled, lambdaa_gnp, cond_numbers, ranks, r_true, loss_ord, base_dir, n_trial, res, symmetric, num_dots=20):
    # Define color palettes for 'scaled' (blue family) and 'gn' (red family)
 
    blue_palette = ['#0d47a1', '#1976d2', '#2196f3', '#64b5f6', '#bbdefb']
    # Shades of red for 'gn' (warm colors)
    red_palette = ['#b71c1c', '#e53935', '#ef5350', '#e57373', '#ffcdd2']


    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'x']  # Different markers for different condition numbers
    linestyles = ['-', '--', '-.', ':']  # Different linestyles for different ranks
    
    # Generate labels for plots
    labels_scaled = [f'scaled, r={r}, cond_n={c}' for c in cond_numbers for r in ranks]
    labels_gn = [f'gn, r={r}, cond_n={c}' for c in cond_numbers for r in ranks]

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
        cond_n, r = label.split(',')[2].strip(), label.split(',')[1].strip()
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
      
    if not res:
        plt.title(f'Matrix Recovery with l{loss_ord} norm, {("BM" if symmetric else "assymetric") + " factor"}, $r^*={r_true}$')
    else:
        plt.title(f'$\\gamma^2 \| \\nabla^\dagger C v \|^2$, $r^*={r_true}$, '
              f'$\\lambda_{{scaled}}={lambdaa_scaled}$, '
              f'$\\lambda_{{gnp}}={lambdaa_gnp}$, loss=l{loss_ord}, $n_{{trial}}=${n_trial}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Distance to M*')
    plt.yscale('log')
    
    lines[1], lines[2] = lines[2], lines[1]

    labels[1], labels[2] = labels[2], labels[1]
    

    lines[5], lines[6] = lines[6], lines[5]

    labels[5], labels[6] = labels[6], labels[5]
    

    
    # Create a custom legend
    plt.legend(handles=lines)
    fig_path = os.path.join(base_dir, f'experiments/{"res" if res else "exp"}_l{loss_ord}_n_trial_{n_trial}.png')
    plt.savefig(fig_path, dpi=300)
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



def create_rip_transform(n, d):
    
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




def matrix_recovery(X0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, lambdaa, method, base_dir, trial):
    
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
    resids = []
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
        
        dampling = lambdaa if lambdaa != 'Liwei' else np.linalg.norm(c(X) - M_star, ord='fro')
        
            
        if method=='scaled':#PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
            try:
                residual = (A(X@X.T) - y_true) if loss_ord==2 else np.sign(A(X@X.T) - y_true)
                try:
                    precondionner_inv =  np.linalg.inv(X.T@X + dampling*np.eye(r,r))
                except:
                    precondionner_inv =  np.eye(r,r)
                
                G = g.reshape(*X.shape)
                preconditionned_G = G @ precondionner_inv
                aux = G @ sqrtm(precondionner_inv) if loss_ord==1 else 'we dont care'
                gamma = (h(c(X)) - 0) / np.sum(np.multiply(aux,aux)) if loss_ord == 1 else 0.000001
            except:
                print("diverged")
                
        elif method=='gnp':
            
            try:
                preconditionned_g, _,_,_ = np.linalg.lstsq(jacob_c.T @ jacob_c + dampling*np.eye(jacob_c.shape[1],jacob_c.shape[1]), g, rcond=-1)
            except:
                preconditionned_g = g #No precondionning 
                
            preconditionned_G = preconditionned_g.reshape(n,r)
            aux = (jacob_c @ preconditionned_g) if loss_ord == 1 else 'we dont care'
            gamma = (h(c(X)) - 0) / np.dot(aux,aux) if loss_ord == 1 else 0.000001
                      

        else:
            raise NotImplementedError
        
        if method == 'gnp':
            resids.append(np.linalg.norm(c(X)  -truncate_svd(c(X), r_true))  )
        else:
            resids.append(np.linalg.norm(c(X)  -truncate_svd(c(X), r_true))  )
        X = X - gamma*preconditionned_G
        
    file_name = f'experiments/exp_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    
    file_name = f'experiments/res_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    
    np.savetxt(full_path, resids, delimiter=',') 
    
    return losses



def matrix_recovery_assymetric(X0, Y0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, lambdaa, method, base_dir, trial):
    
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
    resids = []

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
        
        dampling = lambdaa if lambdaa != 'Liwei' else np.linalg.norm(c(X,Y) - M_star, ord='fro')
        
      
        if method=='gnp':
            
            try:
                preconditionned_g_x, _,_,_ = np.linalg.lstsq(jac_x.T @ jac_x + dampling*np.eye(jac_x.shape[1],jac_x.shape[1]), g_x, rcond=-1)
                preconditionned_g_y, _,_,_ = np.linalg.lstsq(jac_y.T @ jac_y + dampling*np.eye(jac_y.shape[1],jac_y.shape[1]), g_y, rcond=-1)
            except:
                preconditionned_g_x = g_x #No precondionning 
                preconditionned_g_y = g_y
                
            preconditionned_G_x = preconditionned_g_x.reshape(*X.shape)
            preconditionned_G_y = preconditionned_g_y.reshape(*Y.shape)
            aux_x = (jac_x @ preconditionned_g_x) if loss_ord == 1 else 'we dont care'
            aux_y = (jac_y @ preconditionned_g_y) if loss_ord == 1 else 'we dont care'
            gamma = (h(c(X,Y)) - 0) / ( np.dot(aux_x,aux_x) + np.dot(aux_y,aux_y) )  if loss_ord == 1 else 0.000001
          
        
        elif method == 'scaled': #PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
            preconditionner_x = np.linalg.inv(Y.T@Y + dampling* np.eye(X.shape[1]))
            preconditionner_y = np.linalg.inv(X.T@X + dampling* np.eye(Y.shape[1]))
            G_x = g_x.reshape(*X.shape)
            G_y = g_y.reshape(*Y.shape)
            preconditionned_G_x = G_x @ preconditionner_x
            preconditionned_G_y = G_y @ preconditionner_y
            preconditionned_g_x = preconditionned_G_x.reshape(-1)
            preconditionned_g_y = preconditionned_G_y.reshape(-1)
            
            aux_x = G_x @ sqrtm(preconditionner_x) if loss_ord==1 else 'we dont care'
            aux_y = G_y @ sqrtm(preconditionner_y) if loss_ord==1 else 'we dont care'
            gamma = (h(c(X,Y)) - 0) / (np.sum(np.multiply(aux_x,aux_x)) + np.sum(np.multiply(aux_y, aux_y))) if loss_ord == 1 else 0.000001
            
                      

        else:
            raise NotImplementedError
        
        
        X = X - gamma*preconditionned_G_x
        Y = Y - gamma*preconditionned_G_y
        
    file_name = f'experiments/exp_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    
    file_name = f'experiments/res_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    
    np.savetxt(full_path, resids, delimiter=',') 
    
    return losses