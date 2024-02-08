#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:53:34 2024

@author: aglabassi
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_metrics_log_scale(metric_datasets, labels, colors, line_styles, title, xlabel='Iteration', ylabel='Value', logscale=True):
    """
    Plots multiple metric datasets on a logarithmic scale on the same figure.

    Parameters:
    - metric_datasets: List of lists, where each sublist contains metric values for a different experiment.
    - labels: List of labels for the legend, corresponding to each dataset.
    - title: Title for the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 8))  # Adjust size as needed
    for dataset, label, color, line_style in zip(metric_datasets, labels, colors, line_styles):
        plt.plot(dataset, label=label, color=color, linestyle=line_style)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logscale:
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.legend()
    plt.grid(True, which="both", ls="--")  # Improve grid visibility on log scale
    plt.show()  # Display the plot


def matrix_rank_svd(matrix, tol=1e-6):
    # Compute the Singular Value Decomposition
    U, s, V = np.linalg.svd(matrix, full_matrices=False)
    
    # Count the number of singular values larger than the tolerance
    rank = np.sum(s > tol)
    
    return rank


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
    
    
    #random_points = np.random.normal(0, 1, (n, r))*10**-15
    
    return random_points



def generate_matrix_with_condition(n, r, condition_number):
    # Generate a random matrix of dimension n x r
    A = np.random.randn(n, r)
    
    # Perform SVD on the matrix
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Adjust the singular values to achieve the desired condition number
    # Set the largest singular value to the condition number and the smallest to 1
    s = np.linspace(condition_number, 1, min(n, r))
    
    # Construct the diagonal matrix of singular values
    S = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(S, s)
    
    # Reconstruct the matrix with the desired condition number
    A_prime = U @ S @ Vt
    
    return A_prime


def complete_orthogonal_matrix(A):
    """
    Given an orthogonal matrix A of dimension n x r, this function outputs a matrix B
    of dimension n x (n-r) such that the matrix [A, B] is an orthogonal nxn matrix.

    Parameters:
    - A: Orthogonal matrix of shape (n, r)

    Returns:
    - B: Matrix of shape (n, n-r) that, when concatenated with A, forms an orthogonal nxn matrix
    """
    n, r = A.shape
    # Generate a random n x n matrix
    random_matrix = np.random.rand(n, n)
    # Perform QR decomposition on the random matrix
    Q, _ = np.linalg.qr(random_matrix)
    
    # A's columns are already orthonormal. We need to find the orthonormal basis that is orthogonal to A
    # Since Q is orthogonal, its columns form an orthonormal basis of R^n
    # We take the last n-r columns of Q that are orthogonal to the column space of A
    B = Q[:, r:n]
    
    return B

def compact_eigendecomposition(A, tol=1e-10):
    """
    Perform a compact eigendecomposition of a symmetric matrix A, discarding
    eigenvalues that are zero (within a tolerance).

    Parameters:
    - A: A symmetric numpy array of shape (n, n).
    - tol: Tolerance for considering eigenvalues as zero.

    Returns:
    - D: A diagonal matrix (numpy array) of non-zero eigenvalues.
    - U: A matrix (numpy array) of corresponding eigenvectors.
    """
    # Ensure A is symmetric to avoid inaccuracies
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric.")
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Filter out eigenvalues close to zero
    non_zero_indices = np.abs(eigenvalues) > tol
    filtered_eigenvalues = eigenvalues[non_zero_indices]
    filtered_eigenvectors = eigenvectors[:, non_zero_indices]
    
    # Construct diagonal matrix D from filtered eigenvalues
    D = np.diag(filtered_eigenvalues)
    
    # U is already the matrix of filtered eigenvectors
    U = filtered_eigenvectors
    
    return D, U

def compute_iterate_decomposition(X, U_star):
    n, r = X.shape
    r_star = U_star.shape[1]
    
    # Compute S = U_star.T @ X
    S = U_star.T @ X
    
    # Singular Value Decomposition of S to find V
    _, _, VT = np.linalg.svd(S)
    V = VT.T
    
    # Compute U_star complement
    U_star_complement = complete_orthogonal_matrix(U_star)
    
    # Compute N = U_star_complement.T @ X
    N = U_star_complement.T @ X
    
    # Compute A, B, C
    A = U_star @ S
    B = U_star_complement @ N
    
    # Find V complement, which is the orthogonal complement of V in r-dimensional space
    #V_complement = complete_orthogonal_matrix(V[:, :r_star])  # Assuming V's dimensionality
    
    C = U_star_complement @ N
    
    return A, B, C

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






def matrix_recovery(x0, T, h_star, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, loss_ord=1, damek=False):
    
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


    x = x0
    x_flat = x0.reshape(-1)
    losses = []

    losses = []
    error_A = []
    error_B = []
    error_C = []
    error_total = []

    # Pre-compute A*, B*, C*
    #A_star, B_star, C_star = compute_iterate_decomposition(X_true_padded, U_star)
    n,r = x.shape

    for t in range(T):
        
        # Compute the Jacobian of c at x
        jacob_c = jacobian_c(x)
        
        # Compute v from the subdifferential of h at c(x)
        v = subdifferential_h(c(x))
        
        #subdifferential of h(c(x)) w.r.t x
        g = jacob_c.T @ v
        

        if damek:
            tmp = jacob_c.T.shape[0]
            preconditionned_g, _,_,_ = np.linalg.lstsq(jacob_c.T @ jacob_c + lambdaa*np.eye(tmp,tmp), g, rcond=None)
            aux = (jacob_c @ preconditionned_g)
            proj_norm_squared = np.dot(aux , aux)
            preconditionned_g = preconditionned_g.reshape(n,r)
            
        else:
            if loss_ord == 2:
                preconditionned_g = A_adj((A(x@x.T) - y_true))@ x @ np.linalg.inv(x.T@x + lambdaa*np.eye(r,r))
                
            elif loss_ord == 1:
                preconditionned_g = A_adj(( np.sign(A(x@x.T) - y_true)) ) @ x @ np.linalg.inv(x.T@x + lambdaa*np.eye(r,r))
                
            preconditionned_g= preconditionned_g.reshape(-1)
            aux = (jacob_c @ preconditionned_g)
            proj_norm_squared = np.dot(aux , aux)
            preconditionned_g  = preconditionned_g.reshape(n,r)
            
                
                
            
                    
        gamma = (h(c(x)) - h_star) / proj_norm_squared if loss_ord == 1 else 0.000001
        
        #ZHANG FORM
        

        #proj_norm_squared = np.dot(preconditioned_v, preconditioned_v)

        # Update  polyak step size gamma
        
        print('Iteration number: ', t)
        print("r^*=", r_true)
        print("r=", r)
        print('h(c(x)) =', h(c(x)))
        print('---------------------')
        
        # After updating x
        #A_t, B_t, C_t = compute_iterate_decomposition(x, U_star)
        
        # Compute errors and save them
        
        #error_A.append(np.linalg.norm(A_t - A_star, ord='fro'))
        #error_B.append(np.linalg.norm(B_t - B_star, ord='fro'))
        #error_C.append(np.linalg.norm(C_t - C_star, ord='fro'))
        error_total.append(np.linalg.norm(x@x.T - X_true_padded@X_true_padded.T, ord='fro'))
        losses.append(h(c(x)))
        
        x = x - gamma*preconditionned_g
    

    return x, losses, error_A, error_B, error_C, error_total






def zhang(x0, T,A,y_true, lambdaa=0.1):
    pass
    



