#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:17:57 2025

@author: aglabassi
"""

import torch
import os 
import numpy as np
from methods import subgradient_method, LM_subgradient_method, GN_subgradient_method


def generate_point_on_boundary_positive_orthant(p, epsilon):

    radius = torch.norm(p) * epsilon

    
    random_direction = torch.abs(torch.randn_like(p))
    random_direction = random_direction / torch.norm(random_direction)

    new_point = p + radius * random_direction

    return new_point

def generate_difficult_A(m, n, kappa_A=2):
   
    s_values = torch.linspace(1, 1/kappa_A, n)

    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))

    Sigma = torch.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = s_values[i]

    # Construct A = U * Sigma * V^T
    A = U @ Sigma @ V.T

    return A


def setup_experiment(m, n, r_true, kappa, initial_rel_err, device='cpu'):
    """
    Set up the experiment using PyTorch.
    
    Parameters:
      m : int, number of rows of A
      n : int, number of columns of A
      r_true : int, rank for true x_star
      kappa : float, condition number
      device : str, device ('cpu' or 'cuda')
    
    Returns:
      x_star, x0, T, A, b, methods, base_dir
    """

    # Create x_star
    x_star = torch.cat((
        torch.linspace(1, 1/kappa, r_true, device=device),
        torch.zeros(n - r_true, device=device)
    ))


    z0 = generate_point_on_boundary_positive_orthant(x_star**2, initial_rel_err)
    x0 = torch.sqrt(z0)


    A = generate_difficult_A(m, n)

    b = A @ (x_star**2)


    return x_star, x0, A, b



def run_nonegative_least_squares(methods_test, experiment_setups, r_true,
                 m_divided_by_r, device,
                 n_iter, base_dir, initial_rel_err):
    
    outputs = {}
    for r, kappa in experiment_setups:
        
        
        m = m_divided_by_r*r
        n = r
        
        x_star, x0, A, b = setup_experiment(m,n,r_true,kappa, initial_rel_err, device=device)
        
        for method in methods_test:
            
            def stepsize_fc(k,x):
                numerator  = 0.5*( torch.linalg.norm( A@(x*x) - b , ord=2)**2) - 0 
                if method != 'Polyak subgradient':
                    grad = action_nabla_F_transpose_fc(x, subgradient_fc(x))
                    gn_precond_grad = lm_solver(x, 0, grad)
                    denominator = torch.sum(gn_precond_grad * action_nabla_F_transpose_fc(x, 
                                                                                          action_nabla_F_transpose_fc(x, gn_precond_grad )))
                else:
                    denominator = torch.sum( grad*grad )
                return 0.5*numerator/denominator
                
            def damping_fc(k,x):
                return 10**-2 *torch.sqrt( 0.5*( torch.linalg.norm( A@(x*x) - b , ord=2)**2))
            def subgradient_fc(x):
                return A.T@(A@(x*x) - b )
            def action_nabla_F_transpose_fc(x,v):
                return 2*(x*v)
            
            def lm_solver(x,damping, b):
                return b*(1/(4*(x*x)+damping))

            if method == 'Polyak Subgradient':
                xs = subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    x0,
                    n_iter
                )
            elif method == 'Gauss-Newton':
                xs = GN_subgradient_method(
                    stepsize_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    # GN uses zero damping
                    lambda x, g: lm_solver(x, 0.0, g),
                    x0,
                    n_iter
                )
            elif method == 'Levenberg-Marquardt (ours)':
                xs = LM_subgradient_method(
                    stepsize_fc,
                    damping_fc,
                    subgradient_fc,
                    action_nabla_F_transpose_fc,
                    lm_solver,
                    x0,
                    n_iter
                )
            
            
            
            errs = []
            for k,x in enumerate(xs):
                rel_err = torch.norm(x**2 - x_star**2) / torch.norm(x_star**2)
                errs.append(rel_err.item())
                if k % 20 == 0:
                    print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")

            outputs[method] = errs
            fname = f'exphad_{method}_l_{2}_r*={r_true}_r={n}_condn={kappa}_trial_{0}.csv'
            np.savetxt(os.path.join(base_dir, fname), np.array(errs), delimiter=',')
        
               
        return outputs
           