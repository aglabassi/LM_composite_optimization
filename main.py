#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:14:14 2024

@author: aglabassi
"""

# Contents of main.py
from utils import create_rip_transform, generate_matrix_with_condition, compact_eigendecomposition, gen_random_point_in_neighborhood, matrix_recovery, plot_multiple_metrics_log_scale, plot_losses_with_styles
import numpy as np
from gnp import run_gnp


#multiple rank,s one cond number
def experiment_1(r_true, ranks, n, T, cond_number, init_radius_ratio, lambdaa, loss_ord):
        
    d = 10*n*r_true
    
    X_true = generate_matrix_with_condition(n, r_true, cond_number)
    M_true = X_true @ X_true.T
    A,A_adj = create_rip_transform(n, d) 
    y_true = A(M_true)
    
    
    D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
    
    
    radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
    
    
    losses_scaled = []
    losses_gn = []
    for r in ranks:
        
        x0 = gen_random_point_in_neighborhood(X_true, radius, r, r_true)
        
        padding = np.zeros((n, r - r_true))
        X_true_padded = np.hstack((X_true, padding))
        
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=False, loss_ord=loss_ord)
        
        losses_scaled.append(losses)
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=True, loss_ord=loss_ord)
        
        losses_gn.append(losses)
        
    
    
    plot_multiple_metrics_log_scale(losses_scaled+losses_gn, [f'r={r}, scaled' for r in ranks] + [f'r={r}, damek' for r in ranks], 
                                    ['blue']*len(losses_scaled) + ["red"]*len(losses_gn),
                                    [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_scaled))] + [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_gn))], 
                                    f'Loss function for Matrix Recovery, cond_number = {cond_number}, lambda={lambdaa}, loss=l{loss_ord}', xlabel='Iteration', ylabel='Loss', logscale=True)


#multiple cond numbers, single rank
def experiment_2(r_true, cond_numbers, n, T, init_radius_ratio, loss_ord):
    
    d = 10*n*r_true
    
    A,A_adj = create_rip_transform(n, d) 
    
    

    
    losses_scaled = []
    losses_gn = []
    for cond_number in cond_numbers:
        X_true = generate_matrix_with_condition(n, r_true, cond_number)
        D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
        M_true = X_true @ X_true.T
        y_true = A(M_true)
        
        radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
        
        
        x0 = gen_random_point_in_neighborhood(X_true, radius, r_true, r_true)
        
        padding = np.zeros((n, r_true - r_true))
        X_true_padded = np.hstack((X_true, padding))
        
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=False, loss_ord=loss_ord)
        
        losses_scaled.append(losses)
        
        _, losses, errors_A, errors_B, errors_C,_ = matrix_recovery(x0, T, 0, U_star, X_true_padded, lambdaa, r_true, A, A_adj, y_true, damek=True, loss_ord=loss_ord)
        
        losses_gn.append(losses)
    
    #normalize losses: TODO make faster using np
    # normalizor = min( [ loss[0] for loss in losses_gn])
    # for i,l in enumerate(losses_gn):
    #     for j in range(len(l)):
    #         losses_gn[i][j] = np.exp(  np.log(losses_gn[i][j])  - np.log(losses_gn[i][0])  - np.log(normalizor))
    #         losses_scaled[i][j] = np.exp(  np.log(losses_scaled[i][j])  - np.log(losses_scaled[i][0])  - np.log(normalizor))
    
    
        
    
    
    plot_multiple_metrics_log_scale(losses_scaled+losses_gn, [f'cond_n={c}, scaled' for c in cond_numbers] + [f'cond_n={c}, gn' for c in cond_numbers], 
                                    ['blue']*len(losses_scaled) + ["red"]*len(losses_gn),
                                    [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_scaled))] + [f'{"-"*(i+1) if i < 2 else "-."}' for i in range(len(losses_gn))], 
                                    f'Loss function for Matrix Recovery, r_true = {r_true}, lambda=0, loss=l{loss_ord}', xlabel='Iteration', ylabel='Loss', logscale=True)


    
    



#multiple cond numbers and  ranks
def experiment_3(r_true, ranks, cond_numbers, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled):
    
    d = 10*n*r_true
    
    A,A_adj = create_rip_transform(n, d) 
    

    
    losses_scaled = []
    losses_gn = []
    for cond_number in cond_numbers:
        
        X_true = generate_matrix_with_condition(n, r_true, cond_number)
        M_true = X_true @ X_true.T
        A,A_adj = create_rip_transform(n, d) 
        y_true = A(M_true)
        
        
        D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
        
        
        radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
        
        for r in ranks: 
            
            X0 = gen_random_point_in_neighborhood(X_true, radius, r, r_true)
            
            
            losses= matrix_recovery(X0, T, lambdaa_scaled, A, A_adj, y_true, loss_ord, r_true,cond_number, 'scaled')
            
            losses_scaled.append(losses)
            
            losses = matrix_recovery(X0, T, lambdaa_gnp, A, A_adj, y_true, loss_ord, r_true,cond_number, 'gnp')
            
            losses_gn.append(losses)
            
    plot_losses_with_styles(losses_scaled, losses_gn, cond_numbers, ranks, r_true, loss_ord, lambdaa_gnp, lambdaa_scaled)
        
    plt.savefig(f'loss_{loss_ord}.png') 
    
    

    

if __name__ == "__main__":
    
    loss_ord = 1 #L1 or l2 loss. If L2: smooth optimization regime with constant stepsize. If L1: nonsmooth optimization with poliak 

    r_true = 3

    
    
    T = 100
    n = 30
    lambdaa_gnp  = 0
    lambdaa_scaled = 0
    init_radius_ratio = 0.1
    cond_number = 10
    ranks_test = [3, 7]
    cond_numbers_test = [1000]
    
    experiment_3(r_true, ranks_test, cond_numbers_test, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled)
    
    
    #experiment_1(r_true, ranks_test, n, T, cond_number, init_radius_ratio, lambdaa, loss_ord)
      
    
    #experiment_2(r_true, cond_number_tests, n, T, init_radius_ratio, loss_ord)
    
    
