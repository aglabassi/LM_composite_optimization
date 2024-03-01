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
import os
import sys

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



    



#multiple cond numbers and  ranks
def experiment_3(r_true, ranks, cond_numbers, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled, n_trial):
    
    d = 10*n*r_true
    
    A,A_adj = create_rip_transform(n, d) 
    

    losses_scaled_trials = []
    losses_gnp_trials = []
    for trial in range(n_trial):
                        
        losses_scaled_trial = []
        losses_gnp_trial = []
    
        for cond_number in cond_numbers:
            
            X_true = generate_matrix_with_condition(n, r_true, cond_number)
            M_true = X_true @ X_true.T
            A,A_adj = create_rip_transform(n, d) 
            y_true = A(M_true)
            
            D_star_sq, U_star = compact_eigendecomposition(X_true@X_true.T)
            
            
            radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
            
            for r in ranks: 
                
                X0 = gen_random_point_in_neighborhood(X_true, radius, r, r_true)
                
                losses_scaled_trial.append(matrix_recovery(X0, M_true, T,A, A_adj, y_true, loss_ord, r_true,cond_number, lambdaa_scaled, 'scaled'))
                losses_gnp_trial.append(matrix_recovery(X0, M_true,T, A, A_adj, y_true, loss_ord, r_true,cond_number, lambdaa_gnp, 'gnp'))
                
            losses_scaled_trials.append(losses_scaled_trial)
            losses_gnp_trials.append(losses_gnp_trial)

    
    losses_scaled = np.mean(np.array(losses_scaled_trials), axis=0)
    losses_gnp = np.mean(np.array(losses_gnp_trials), axis=0)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    plot_losses_with_styles(losses_scaled, losses_gnp, lambdaa_scaled, lambdaa_gnp, cond_numbers, ranks, r_true, loss_ord, base_dir)
    

    

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        try:
            loss_ord = int(sys.argv[1])  # Convert first command-line argument to integer
        except ValueError:
            print("Please provide an integer value for loss_ord.")
            sys.exit(1)
    else:
        print("No loss_ord provided, using default value of 1.")
        loss_ord = 1  # Default value if not provided

        
 
    r_true = 3
    np.random.seed(seed=42)  
    
    n_trial = 10
    T = 1000
    n = 30
    lambdaa_gnp  = 'Liwei'
    lambdaa_scaled = 'Liwei'
    init_radius_ratio = 0.1
    cond_number = 10
    ranks_test = [3,20]
    cond_numbers_test = [1,1000]
    
    experiment_3(r_true, ranks_test, cond_numbers_test, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled, n_trial)
    
    
 
    
