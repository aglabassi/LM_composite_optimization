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



def sample_symmetric_matrix_on_boundary(M, eps):
    """
    Given a symmetric matrix M (n x n) of rank r and a scalar eps,
    generate a symmetric matrix A (of rank r) such that:
        ||A - M||_F = eps * ||M||_F,
    with A chosen "uniformly at random" from matrices of the form M + U X U^T,
    where X is a symmetric r x r matrix uniformly distributed on the Frobenius sphere.
    
    Parameters:
      M   : numpy.ndarray, shape (n,n), symmetric of rank r.
      eps : float, the factor in the norm condition.
    
    Returns:
      A : numpy.ndarray, shape (n,n), symmetric, of rank r.
    """
    # Compute the eigendecomposition of M.
    # (Using eigh since M is symmetric.)
    eigvals, eigvecs = np.linalg.eigh(M)
    
    # Select the nonzero eigenvalues (we use a tolerance to decide)
    tol = 1e-10
    nonzero_indices = np.where(np.abs(eigvals) > tol)[0]
    r = len(nonzero_indices)
    U = eigvecs[:, nonzero_indices]
    Sigma = np.diag(eigvals[nonzero_indices])
    
    # Determine the target radius for the perturbation.
    norm_M = np.linalg.norm(M, 'fro')  # equals np.linalg.norm(Sigma, 'fro')
    R = eps * norm_M
    
    # The space of symmetric r x r matrices has dimension d = r(r+1)/2.
    d = r * (r + 1) // 2
    
    # Generate d independent standard normals.
    x = np.random.randn(d)
    
    # Normalize the vector to have norm R.
    x = (R / np.linalg.norm(x)) * x
    
    # Map the vector x to a symmetric r x r matrix X.
    X = np.zeros((r, r))
    idx = 0
    for i in range(r):
        for j in range(i, r):
            # Fill the upper triangle (and mirror it).
            X[i, j] = x[idx]
            if i != j:
                X[j, i] = x[idx]
            idx += 1
    
    # Form the perturbation Delta in the range of M.
    Delta = U @ X @ U.T
    
    # Create A by adding the perturbation to M.
    A = M + Delta
    return A


def trial_execution_matrix(trials, n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric=True, identity=False, corr_factor=0,
                           gamma=10**-8, lambda_=0.00001, q=0.97, geom_decay=False):
    
    for trial in trials:
        A, A_adj = create_rip_transform(n, d, identity)
    
        for r, cond_number in keys:
            X_true = generate_matrix_with_condition(n, r_true, cond_number)
            Y_true = generate_matrix_with_condition(n, r_true, cond_number)
            
            if symmetric:
                M_true =  X_true @ X_true.T
            else:
                M_true = X_true @ Y_true.T
                
            M_corrupted = np.random.rand(n,n)
            y_corrupted = A(M_corrupted)
                
            y_true = A(M_true)
            num_ones = int(y_true.shape[0]*corr_factor)
            mask_indices = np.random.choice(y_true.shape[0], size=num_ones, replace=False)
            mask = np.zeros(y_true.shape[0])
            mask[mask_indices] = 1

    
            
            
        

            #y_true = y_true + np.linalg.norm(y_true)*np.random.normal(size=y_true.shape[0])*mask
            y_true = (1-mask)*y_true + mask*y_corrupted
  
            pert = np.random.rand(*M_true.shape)
            if symmetric:
                pert = (pert + pert.T) /2
                
            Z0 = sample_symmetric_matrix_on_boundary(M_true, init_radius_ratio)
            
    
            U, Sigma, VT = np.linalg.svd(Z0)
            U_r = U[:, :r]
            Sigma_r = Sigma[:r]
            VT_r = VT[:r, :]
            
            # Compute Sigma_r^{1/2}
            Sigma_r_sqrt = np.sqrt(Sigma_r)
            
            # Compute X and Y  
            X0 = U_r @ np.diag(Sigma_r_sqrt)
            Y0 = (VT_r.T) @ np.diag(Sigma_r_sqrt)
            #X0 = 10**(-10)*np.random.randn(n,r) random init.
            outputs = dict()
            for method in methods:
                if symmetric:
                    losses = matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, 
                                             r_true, cond_number, method, base_dir, trial, gamma_init=gamma, 
                                             damping_init=lambda_, q=q, geom_decay=geom_decay)
                else:
                    losses = matrix_recovery_assymetric(X0, Y0, X_true, Y_true, M_true, T, A, A_adj, y_true, loss_ord, 
                                                        r_true, cond_number, method, base_dir, trial, gamma_init=gamma, 
                                                        damping_init=lambda_, q=q, geom_decay=geom_decay)
                
                outputs[method]= losses 
    return outputs



    
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
    cond_number, method, base_dir, trial, gamma_init=10**-8, damping_init=10**-5, q=0.98, geom_decay=False):
    """
    If restarted=False, run the original (non-restarted) method you provided.
    If restarted=True, run a 'Restarted Levenberg-Marquardt Subgradient Method'.
    """
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
        if t%20==0:
            print("symmetric")
            print(f'Iteration number :  {t}')
            print(f'Method           :  {method}')
            print(f'Cond. number     :  {cond_number}')
            print(f"r*, r            :  {(r_true, r)}")
            print(np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star))
            val_cX = h(c(X))
            print(f'h(c(X)) = {"(DIVERGE)" if (np.isnan(val_cX) or val_cX == np.inf ) else val_cX}')
            print('---------------------')

        val_cX = h(c(X))
        if np.isnan(val_cX) or val_cX == np.inf or \
           np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star) > 1e4:
            # Indicate divergence in the losses
            losses += [1]*(n_iter - len(losses))
            break
        elif np.linalg.norm(c(X) - M_star)/np.linalg.norm(M_star) <= 1e-2000:
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
                damping =  damping_init*q**t  if geom_decay else  np.sqrt( h(c(X))) /4000 if loss_ord == 2 else h(c(X))/100000
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
        gamma = gamma_init*(q**t)  if geom_decay else gamma#geometrically decaying stepsize
        #gamma = 0.1*gamma
        X = X - gamma*preconditionned_G

    # Save results
    file_name = f'./expbm_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
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
                               gamma_init=0.001, damping_init=0.0001, q=0.8, geom_decay=False):
    
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
            if geom_decay:
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
        gamma = gamma_init*q**t if geom_decay else gamma
        X = X - gamma*preconditionned_G_x
        Y = Y - gamma*preconditionned_G_y
        
        if match_method_pattern(method, prefix='OPSA')[0]:
            X,Y = rebalance(X,Y)
        
        
    file_name = f'expasymmetric_{method}_l_{loss_ord}_r*={r_true}_r={X.shape[1]}_condn={cond_number}_trial_{trial}.csv'
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

