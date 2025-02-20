#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:34:20 2025

@author: aglabassi
"""


import torch
import os 
import numpy as np #use to save csv's 
from preconditioners import LM_preconditioner, scaled_preconditioner
from utils import LinearMeasurementOperator, generate_data_and_initialize, \
    compute_gradient, compute_stepsize_and_damping, cg_solve, update_factors, split

def test_methods(methods_test, experiment_setups, n1, n2, n3, r_true, m, identity, device,
                n_iter, base_dir, loss_ord, initial_relative_error, symmetric,
                tensor=True, corr_level=0, geom_decay=False, q=0.97, lambda_=10**-5, gamma=10**-8):
    """
    Runs the various descent methods.
    
    Depending on the Boolean flag "tensor", the reconstruction, initialization,
    gradient and operator routines are selected to handle either CP–tensor or
    matrix factorization.
    """
    # Initialize measurement operator.
    # For CP–tensor: use (n1,n2,n3); for matrix factorization:
    #   symmetric: use (n1,n1) and asymmetric: use (n1,n2)
 
    measurement_operator = LinearMeasurementOperator(n1, n2, n3, m, device,
                                                     identity=identity,
                                                     tensor=tensor)

    
    outputs = dict()
    for experiment_setup in experiment_setups:
        r, kappa = experiment_setup
        print("=" * 80)
        print(f"Experiment Setup: r = {r}, kappa = {kappa}")
        print("=" * 80)
        
        T_star, y_observed, factors, sizes = generate_data_and_initialize(
                measurement_operator=measurement_operator,
                n1=n1,
                r_true=r_true,
                r=r,
                n2=n2,
                n3=n3,
                device=device,
                kappa=kappa,
                corr_level=corr_level,
                symmetric=symmetric,
                tensor=tensor,
                initial_relative_error=initial_relative_error)
        
        if not tensor:
            X0, Y0 = factors
        else:
            X0, Y0, Z0 = factors

        for method in methods_test:
            print(f"\n{'-' * 80}\nStarting method: {method}\n{'-' * 80}")
            # Reset factors for each method.
            X = X0.clone()
            Y = Y0.clone()
            if tensor:
                Z = Z0.clone()

        
            errs = []

            for k in range(n_iter):
                # Reconstruct the model.
                if tensor:
                    T = torch.einsum('ir,jr,kr->ijk',X, Y, Z)
                else:
                    T = X @ Y.T

                err = torch.norm(T - T_star)
                rel_err = err / torch.norm(T_star)
                if k % 20 == 0:
                    print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")
                if rel_err < 1e-14:
                    errs += [1e-15 for _ in range(k, n_iter)]
                    break
                errs.append(rel_err.item())

                # Compute the residual.
                residual = measurement_operator.A(T) - y_observed

                # Compute subgradient based on loss_ord.
                if loss_ord == 1:
                    subgradient_h = measurement_operator.A_adj(torch.sign(residual)).view(-1)
                    h_c_x = torch.sum(torch.abs(residual)).item()
                elif loss_ord == 0.5:
                    subgradient_h = measurement_operator.A_adj(residual / torch.norm(residual)).view(-1)
                    h_c_x = torch.norm(residual).item()
                elif loss_ord == 2:
                    subgradient_h = measurement_operator.A_adj(residual).view(-1)
                    h_c_x = 0.5 * (torch.norm(residual) ** 2).item()

                grad = compute_gradient(X,
                                        Y,
                                        Z if tensor else None,
                                        subgradient_h,  
                                        n1,
                                        n2,
                                        n3,
                                        symmetric=symmetric,
                                        tensor=tensor)

                stepsize, damping = compute_stepsize_and_damping(
                                        method,
                                        grad,
                                        subgradient_h,
                                        h_c_x,
                                        loss_ord,
                                        symmetric,
                                        geom_decay=geom_decay,
                                        q=q, lambda_=lambda_, gamma=gamma, k=k)
       
                factors = [ X,Y,Z ] if tensor else [X,Y]
                if method in ['Gauss-Newton', 'Levenberg-Marquardt (ours)']:
                    operator_fn = lambda x: LM_preconditioner(x, factors, symmetric, tensor=tensor) 
                elif method in ['Precond. gradient', 'Scaled gradient($\lambda=10^{-8}$)', 'OPSA($\lambda=10^{-8}$)']:
                    operator_fn = lambda x: scaled_preconditioner(x, factors, symmetric, tensor=tensor)
                elif method in ['Gradient descent', 'Subgradient descent']:
                    operator_fn = lambda x: x
                else:
                    raise NotImplementedError()
                            
                preconditioned_grad = cg_solve(operator_fn, grad, damping)
                    
                X, Y, Z = update_factors( X, Y, (Z if tensor else None), 
                                        preconditioned_grad, 
                                        stepsize, 
                                        sizes, 
                                        split, 
                                        symmetric=symmetric, 
                                        tensor=tensor)
                
            
        

            print(f"Method '{method}' completed after {k+1} iterations with final relative error: {errs[-1]:.3e}\n")
            file_name = f'exp{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}_{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_0.csv'
            full_path = os.path.join(base_dir, file_name)
            np.savetxt(full_path, np.array(errs), delimiter=',')
            outputs[method] = errs
    return outputs
