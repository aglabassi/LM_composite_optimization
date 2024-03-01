#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:14:14 2024

@author: aglabassi
"""

# Contents of main.py
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles
import numpy as np
import os
import sys

    



#multiple cond numbers and  ranks
def experiment(r_true, ranks, cond_numbers, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled, n_trial):
    
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
        loss_ord = 1  # Default value if not provided
        print(f"No loss_ord provided, using default value of {loss_ord}.")

        
 
    r_true = 3
    np.random.seed(seed=42)  
    
    n_trial = 10
    T = 1000
    n = 30
    lambdaa_gnp  = 'Liwei'
    lambdaa_scaled = 'Liwei'
    init_radius_ratio = 0.1
    ranks_test = [3,20]
    cond_numbers_test = [1,1000]
    
    experiment(r_true, ranks_test, cond_numbers_test, n, T, init_radius_ratio, loss_ord, lambdaa_gnp, lambdaa_scaled, n_trial)
    
    
 
    
