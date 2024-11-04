# -*- coding: utf-8 -*-
# Author: Abdel
# Implementation of method in "Fast and Provable Tensor Robust Principal Component Analysis via Scaled Gradient Descent"

import torch
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from tensor_RPCA_Theirs import rpca
from tensor_RPCA_Ours import tensor_recovery
import matplotlib.pyplot as plt
from utils import random_perturbation
import os
from utils import collect_compute_mean, plot_losses_with_styles

tl.set_backend('pytorch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_tensor(n, r_true, r, kappa):
    U1 = torch.linalg.svd(torch.rand(n, n, device=device).double())[0][:, :r_true]
    U2 = torch.linalg.svd(torch.rand(n, n, device=device).double())[0][:, :r_true]
    U3 = torch.linalg.svd(torch.rand(n, n, device=device).double())[0][:, :r_true]
        
        # Adjust singular values to set the condition number
    def set_condition_number(U, kappa):
        # Create singular values spanning from 1 to 1/kappa
        singular_values = torch.linspace(1, 1/kappa, U.shape[1], device=device, dtype=torch.double)
        U_adjusted = U @ torch.diag(singular_values)
        return U_adjusted
    
    # Apply condition number adjustment
    U1 = set_condition_number(U1, kappa)
    U2 = set_condition_number(U2, kappa)
    U3 = set_condition_number(U3, kappa)
        
    core_G = torch.zeros((r_true, r_true, r_true), device=device).double()
    indices = torch.arange(r_true, device=device).long()
    core_G[indices, indices, indices] = 1
    
    mu = (n / r_true) * max(torch.max(torch.norm(U1, dim=0, p=float('inf'))),
                            torch.max(torch.norm(U2, dim=0, p=float('inf'))),
                            torch.max(torch.norm(U3, dim=0, p=float('inf'))))
    
    padding = (0, r - r_true, 0, r - r_true, 0, r - r_true)
    core_G_extended = torch.nn.functional.pad(core_G, padding)
    
    U1_extended = torch.cat((U1, torch.zeros(n, r - r_true).double()), dim=1)
    U2_extended = torch.cat((U2, torch.zeros(n, r - r_true).double()), dim=1)
    U3_extended = torch.cat((U3, torch.zeros(n, r - r_true).double()), dim=1)
    
    indices_r = torch.arange(r, device=device).long() 
    core_G_extended[indices_r, indices_r, indices_r] = 1
    
    return mu, core_G_extended, [U1_extended, U2_extended, U3_extended]


def matrixise(T, mode):
    return tl.unfold(T, mode).double()

def generate_random_mask(m, n, p, alpha):
    total_elements = m * n * p
    n_ones = int(alpha * total_elements)
    flat_mask = torch.zeros(total_elements, dtype=torch.int, device=device)
    flat_mask[:n_ones] = 1
    flat_mask = flat_mask[torch.randperm(total_elements, device=device)]
    return flat_mask.reshape(m, n, p).double()

def generate_uniform_random(a, b, size):
    return a + (b - a) * torch.rand(size, device=device).double()

def min_singular_value(matrix, eps=1e-5):
    singular_values = torch.linalg.svdvals(matrix).double()
    significant_values = singular_values[singular_values > eps]
    if len(significant_values) == 0:
        return 0
    return significant_values.min()


class TensorMeasurementOperator:
    def __init__(self, n1, n2, n3, m, identity=False):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.m = m
        self.identity = identity
        
        # Generate m measurement tensors Ai ~ N(0, 1/m)
        self.A_tensors = torch.randn(m, n1, n2, n3, device=device).double() / torch.sqrt(torch.tensor(m, dtype=torch.float64))

    def A(self, X):
        if self.identity:
            # Flatten X and return
            return X.flatten()
        else:
            # Apply operator A(X⋆) = {⟨Ai, X ⟩}m
            inner_products = torch.einsum('ijkl,jkl->i', self.A_tensors, X)
            return inner_products

    def A_adj(self, y):
        if self.identity:
            # Reshape y back into the original tensor shape
            return y.reshape(self.n1, self.n2, self.n3)
        else:
            # Compute the adjoint A*(y) = sum(y_i * Ai)
            y_expanded = y.reshape(self.m, 1, 1, 1)  # Expand y to match dimensions
            adjoint = torch.sum(y_expanded * self.A_tensors, dim=0)
            return adjoint

def run_methods(methods, keys, n, r_true, target_d, identity, device, 
                stepsize, decay_constant, n_iter, spectral_init, 
                base_dir, loss_ord, radius_init, fix_G=False):
    for method in methods:
        for r, kappa in keys:
            
            # Generate tensor
            mu, G, factors = generate_tensor(n, r_true, r, kappa)
            T_true = tl.tucker_to_tensor((G, factors))
            
            # Calculate thresholds
            treshold_0 = 1.5 * T_true.abs().max().item()
            treshold_1 = (2 * torch.sqrt(mu * r / n))**3 * min(
                min_singular_value(matrixise(T_true, 0)),
                min_singular_value(matrixise(T_true, 1)),
                min_singular_value(matrixise(T_true, 2))
            )
            
            # Calculate corruption factor
            corruption_factor = 100 / torch.sqrt(kappa * ((mu * r)**3))
            corruption_factor = 0
            corr_scaler = 0
            
            # Create corrupted tensor
            T_true_corr = T_true + corr_scaler * torch.mul(
                generate_random_mask(n, n, n, corruption_factor),  
                generate_uniform_random(-1 * T_true.abs().mean().item(), T_true.abs().mean().item(), (n, n, n))
            )
            
            # Create measurement operator
            measurement_operator = TensorMeasurementOperator(n, n, n, target_d, identity=identity)
            
            # Initialize G0 and factors0 based on fix_G flag
            if fix_G:
                G0 = G.clone()
                factors0 = parafac(T_true + random_perturbation((n, n, n), radius_init).to(device), rank=r)[1]
            else:
                G0, factors0 = tucker(T_true + random_perturbation((n, n, n), radius_init).to(device), rank=[r, r, r])
            
            # Run selected method
            if method == 'scaled_GD':
                rpca(G, factors, G0, factors0, T_true, T_true_corr, measurement_operator, r_true, kappa, 
                     [r, r, r], treshold_0, treshold_1, stepsize, decay_constant, n_iter, 10**-10, device, 
                     spectral_init, base_dir, loss_ord, perturb=radius_init, fix_G=fix_G)
            else:
                tensor_recovery(method, G, factors, G0, factors0, T_true, T_true_corr, measurement_operator, 
                                r_true, kappa, [r, r, r], treshold_0, n_iter, spectral_init, base_dir, loss_ord, 
                                perturb=radius_init, fix_G=fix_G)

# Updated variables with specified values
methods = ['gnp','subGD']
keys = [ (2,1), (2,10)]
n = 10
r_true = 2
target_d = n * r_true * 10
identity = False
device = 'cpu'
stepsize = 0.4
decay_constant = 1 - 0.45 * stepsize
n_iter = 1000
spectral_init = False
base_dir = os.path.dirname(os.path.abspath(__file__))
loss_ord = 1
radius_init = 0.00001
fix_G = True

# Call the function
run_methods(methods, keys, n, r_true, target_d, identity, device, 
            stepsize, decay_constant, n_iter, spectral_init, base_dir, 
            loss_ord, radius_init, fix_G)

errs = collect_compute_mean(keys, loss_ord, r_true,False, methods, 'tensor')
plot_losses_with_styles(errs, r_true, loss_ord, base_dir,  ('CP' if fix_G else 'Tucker'))





