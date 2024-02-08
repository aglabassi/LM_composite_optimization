#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:14:14 2024

@author: aglabassi
"""

# Contents of main.py
from utils import create_rip_transform, generate_matrix_with_condition, compact_eigendecomposition, gen_random_point_in_neighborhood, matrix_recovery, plot_multiple_metrics_log_scale
import numpy as np


#P
def experiment_1(r_true, ranks, n, T, cond_number, init_radius_ratio, lambdaa, loss_ord):
        
    d = 10*n*r_true
    
    X_true = generate_matrix_with_condition(n, r_true, cond_number)
    M_true = X_true @ X_true.T
    A,A_adj = create_rip_transform(n, d) 
    y_true = A(M_true)
    
    
    D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
    
    
    radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
    
    
    losses_scaled = []
    losses_gnp = []
    for r in ranks:
        
        x0 = gen_random_point_in_neighborhood(X_true, radius, r, r_true)
        
        padding = np.zeros((n, r - r_true))
        X_true_padded = np.hstack((X_true, padding))
        
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=False, loss_ord=2)
        
        losses_scaled.append(losses)
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=True, loss_ord=2)
        
        losses_gnp.append(losses)
        
    
    
    plot_multiple_metrics_log_scale(losses_scaled+losses_gnp, [f'r={r}, scaled' for r in ranks] + [f'r={r}, gnp' for r in ranks], 
                                    ['blue']*len(losses_scaled) + ["red"]*len(losses_gnp),
                                    [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_scaled))] + [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_gnp))], 
                                    f'Loss function for Matrix Recovery, cond_number = {cond_number}, lambda={lambdaa}, loss=l{loss_ord}', xlabel='Iteration', ylabel='Loss', logscale=True)



def experiment_2(r_true, cond_numbers, n, T, init_radius_ratio, loss_ord):
    
    d = 10*n*r_true
    
    A,A_adj = create_rip_transform(n, d) 
    
    

    
    losses_scaled = []
    losses_gnp = []
    for cond_number in cond_numbers:
        X_true = generate_matrix_with_condition(n, r_true, cond_number)
        D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
        M_true = X_true @ X_true.T
        y_true = A(M_true)
        
        radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
        
        
        x0 = gen_random_point_in_neighborhood(X_true, radius, r_true, r_true)
        
        padding = np.zeros((n, r_true - r_true))
        X_true_padded = np.hstack((X_true, padding))
        
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=False, loss_ord=2)
        
        losses_scaled.append(losses)
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=True, loss_ord=2)
        
        losses_gnp.append(losses)
    
    #normalize losses: TODO make faster using np
    # normalizor = min( [ loss[0] for loss in losses_gnp])
    # for i,l in enumerate(losses_gnp):
    #     for j in range(len(l)):
    #         losses_gnp[i][j] = np.exp(  np.log(losses_gnp[i][j])  - np.log(losses_gnp[i][0])  - np.log(normalizor))
    #         losses_scaled[i][j] = np.exp(  np.log(losses_scaled[i][j])  - np.log(losses_scaled[i][0])  - np.log(normalizor))
    
    
        
    
    
    plot_multiple_metrics_log_scale(losses_scaled+losses_gnp, [f'cond_n={c}, scaled' for c in cond_numbers] + [f'cond_n={c}, gnp' for c in cond_numbers], 
                                    ['blue']*len(losses_scaled) + ["red"]*len(losses_gnp),
                                    [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_scaled))] + [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_gnp))], 
                                    f'Loss function for Matrix Recovery, r_true = {r_true}, lambda=0, loss=l{loss_ord}', xlabel='Iteration', ylabel='Loss', logscale=True)


    

if __name__ == "__main__":
    
    loss_ord = 1

    r_true = 3

    
    
    T = 200
    n = 30
    lambdaa  = 0.000000001
    init_radius_ratio = 0.1
    cond_number = 10
    ranks_test = [3, 15]
    cond_number_tests = [1,100,1000]
    
    experiment_1(r_true, ranks_test, n, T, cond_number, init_radius_ratio, lambdaa, loss_ord)
      
    
    #experiment_2(r_true, cond_number_tests, n, T, init_radius_ratio, loss_ord)
    
    
