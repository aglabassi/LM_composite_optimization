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

    for rank, cond_number in keys:
        for method in methods:
            tensor = '' if problem != 'tensor' else "tensor"
            file_pattern = f"experiments/{('res' if res else 'exp') + tensor}_{method}_l_{loss_ord}_r*={r_true}_r={rank}_condn={cond_number}_trial_*.csv"
            file_list = glob.glob(file_pattern)
            data_list = []
            
            # Read each file and append its data to the data_list
            for file in file_list:
                data_list.append(np.loadtxt(file, delimiter=','))  # Assume the CSV is correctly formatted for np.loadtxt
            
            # Convert the list of arrays into a 2D numpy array for easier manipulation
            data_array = np.array(data_list)
            
            # Compute the mean across all trials (rows) for each experiment
            mean_values = np.mean(data_array, axis=0)
            losses[method][(rank, cond_number)]  = mean_values
   
    return losses



def trial_execution_matrix(trials, n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric=True):
    
    for trial in trials:
        A, A_adj = create_rip_transform(n, d)
    
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
                    matrix_recovery_assymetric(X0, Y0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial)
                    
    return 'we dont care'






def plot_losses_with_styles(losses, r_true, loss_ord, base_dir, problem, num_dots=20):
    # Define color palettes for 'scaled' (blue family) and 'gn' (red family)
    
    methods = losses.keys()
    for m in methods:
        for m2 in methods:
            assert losses[m].keys() == losses[m2].keys()
    
    colors_list = [
        ['#0d47a1', '#1976d2', '#2196f3', '#64b5f6', '#bbdefb'],  # Blue palette
        ['#b71c1c', '#e53935', '#ef5350', '#e57373', '#ffcdd2'],  # Red palette
        ['#4a148c', '#8e24aa', '#ab47bc', '#ce93d8', '#e1bee7'],  # Purple palette
        ['#1b5e20', '#388e3c', '#4caf50', '#81c784', '#c8e6c9'],  # Green palette
        ['#f57f17', '#fbc02d', '#ffeb3b', '#fff176', '#fff9c4']   # Yellow palette
    ]
    
    assert len(colors_list) >= len(methods) #enough colors for each method


    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'x']  # Different markers for different condition numbers
    linestyles = ['-', '--', '-.', ':']  # Different linestyles for different ranks
    
    



    # Prepare the plot
    plt.figure(figsize=(10, 6))
    
    keys = losses[list(methods)[0]].keys()
    
    rs = []
    cs = []
    for k in keys:
        r,c = k
        if r not in rs:
            rs.append(r)
        if c not in cs:
            cs.append(c)
    
    for idx_m, method in enumerate(methods):
        for idx, k in enumerate(keys):
            r_index = rs.index(k[0])
            c_index = cs.index(k[1])
            errs = losses[method][k]
            
            color = colors_list[idx_m][idx]

            
            marker = markers[c_index]
            linestyle = linestyles[r_index]
            indices = np.arange(0, len(errs), num_dots)
            plt.plot(
                indices,
                errs[indices],
                linestyle=linestyle,
                color=color,
                marker=marker,
                label=(
                    f'{method} {", overparam." if k[0] > r_true else ""} '
                    f'{", ill-conditioned" if k[1] > 1 else ""}'
                )
            )
        
          
    object_ = 'Matrix' if problem == 'Burer-Monteiro' or problem == 'Left-Right' else 'Tensor'
    plt.title(rf'{ object_ } sensing with $\ell_{loss_ord}-norm$, {problem}')
    
    plt.xlabel('Iteration')
    plt.ylabel(r'Distance to $M^\ast$')
    plt.yscale('log')

    
    # Create a custom legend
    plt.legend()
    fig_path = os.path.join(base_dir, f'experiments/exp_l{loss_ord}.png')
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


# def create_rip_transform(n, d):
#     # Create an identity matrix for the flattening operation
#     transformation_matrix = np.eye(n*n)
#     adjoint_transformation_matrix = transformation_matrix.T

#     def transform(matrix):
#         # Flatten the matrix directly
#         return np.dot(transformation_matrix, matrix.reshape(-1))

#     def adjoint_transform(vector):
#         # Reshape the vector back to the original matrix shape
#         v = np.dot(adjoint_transformation_matrix, vector)
#         return v.reshape(n, n)

#     return transform, adjoint_transform



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
        
        
        
        
        
            
        if method == 'scaled_GD' or method == 'scaled_subGD' or method == 'pred_GD'  or method == 'scaled_GD_lambda': #PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
        
           
            if method == 'scaled_GD' or 'scaled_subGD':
                dampling = 0
            elif method == 'pred_GD':
                dampling = np.linalg.norm(c(X) - M_star, ord='fro')
            elif method == 'scaled_GD_lambda':
                dampling = 10**-16
            
            precondionner_inv =   np.linalg.inv( X.T@X + dampling* np.eye(r) )  #(np.inv()) for scaled
            
            
            G = g.reshape(*X.shape)
            preconditionned_G = G @ precondionner_inv
            aux = G @ sqrtm(precondionner_inv)
            gamma = (h(c(X)) - 0) / (np.sum(np.multiply(aux,aux))) if method == 'scaled_subGD' else  0.000001
            #constant 
            #gamma = 0.000001
            
               
        elif method=='gnp':
            
            try:
                dampling = np.linalg.norm(c(X) - M_star, ord='fro')
                preconditionned_g, _,_,_ = np.linalg.lstsq(jacob_c.T @ jacob_c + dampling*np.eye(jacob_c.shape[1],jacob_c.shape[1]), g, rcond=-1)
            except:
                preconditionned_g = g #No precondionning 
                
            preconditionned_G = preconditionned_g.reshape(n,r)
            gamma = 2*(h(c(X)) - 0) / np.dot(v,v)
        elif method == 'subGD':
            preconditionned_G  = g.reshape(n,r)
            gamma = (h(c(X)) - 0) / np.sum(np.multiply(g,g))
            

        else:
            raise NotImplementedError
        

        X = X - gamma*preconditionned_G
    
    file_name = f'experiments/exp_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
    
    
    return losses



def matrix_recovery_assymetric(X0, Y0, M_star, n_iter, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial):
    
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
        
       
        
      
        if method=='gnp':
            dampling = np.linalg.norm(c(X,Y) - M_star, ord='fro')
            
            try:
                
                preconditionned_g_x, _,_,_ = np.linalg.lstsq(jac_x.T @ jac_x + dampling*np.eye(jac_x.shape[1],jac_x.shape[1]), g_x, rcond=-1)
                preconditionned_g_y, _,_,_ = np.linalg.lstsq(jac_y.T @ jac_y + dampling*np.eye(jac_y.shape[1],jac_y.shape[1]), g_y, rcond=-1)
            except:
                preconditionned_g_x = g_x #No precondionning 
                preconditionned_g_y = g_y
                
            preconditionned_G_x = preconditionned_g_x.reshape(*X.shape)
            preconditionned_G_y = preconditionned_g_y.reshape(*Y.shape)
            gamma = 1.5*(h(c(X,Y)) - 0) / ( np.dot(v,v) ) 
          
        
        elif method == 'scaled_GD' or method == 'scaled_subGD' or method == 'pred_GD'  or method == 'scaled_GD_lambda': #PAPER: Low-Rank Matrix Recovery with Scaled Subgradient Methods
            
           
           if method == 'scaled_GD' or 'scaled_subGD':
               dampling = 0
           elif method == 'pred_GD':
               dampling = np.linalg.norm(c(X,Y) - M_star, ord='fro')
           elif method == 'scaled_GD_lambda':
               dampling = 10**-16
           
           preconditionner_x = np.linalg.inv(Y.T@Y + dampling* np.eye(X.shape[1]))
           preconditionner_y = np.linalg.inv(X.T@X + dampling* np.eye(Y.shape[1]))
           dampling
           G_x = g_x.reshape(*X.shape)
           G_y = g_y.reshape(*Y.shape)
           preconditionned_G_x = G_x @ preconditionner_x
           preconditionned_G_y = G_y @ preconditionner_y
           preconditionned_g_x = preconditionned_G_x.reshape(-1)
           preconditionned_g_y = preconditionned_G_y.reshape(-1)
           
           aux_x = G_x @ sqrtm(preconditionner_x) 
           aux_y = G_y @ sqrtm(preconditionner_y) 
           gamma = (h(c(X,Y)) - 0) / (np.sum(np.multiply(aux_x,aux_x)) + np.sum(np.multiply(aux_y, aux_y))) if method == 'scaled_subGD' else  0.000001
        
        elif method == 'subGD': 
            preconditionned_G_x = g_x.reshape(*X.shape)
            preconditionned_G_y = g_y.reshape(*Y.shape)
            gamma = (h(c(X,Y)) - 0) /  np.sum(np.multiply( g_x, g_x)) + np.sum(np.multiply( g_y, g_y)) 
        else:
            raise NotImplementedError
        
        
        X = X - gamma*preconditionned_G_x
        Y = Y - gamma*preconditionned_G_y
        
    file_name = f'experiments/exp_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, losses, delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
    
    
    return losses



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




