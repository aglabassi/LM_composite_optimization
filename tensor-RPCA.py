# -*- coding: utf-8 -*-
# Author: Abdel
# Implementation of method in "Fast and Provable Tensor Robust Principal Component Analysis via Scaled Gradient Descent"

import torch
import tensorly as tl
from tensor_RPCA_Theirs import rpca
from tensor_RPCA_Ours import rpca_ours
import matplotlib.pyplot as plt

tl.set_backend('pytorch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_tensor(n, r_true, r, kappa):
    
    
    
    U1 = torch.linalg.svd(torch.rand(n, n, device=device))[0][:, :r_true]
    U2 = torch.linalg.svd(torch.rand(n, n, device=device))[0][:, :r_true]
    U3 = torch.linalg.svd(torch.rand(n, n, device=device))[0][:, :r_true]
    core_G = torch.zeros((r_true, r_true, r_true), device=device)
    indices = torch.arange(r_true, device=device)
    core_G[indices, indices, indices] = kappa ** (-indices / (r_true - 1))
    
    mu = (n/r_true)*max(torch.max(torch.norm(U1, dim=0, p=float('inf'))),
                   torch.max(torch.norm(U2, dim=0, p=float('inf'))),
                   torch.max(torch.norm(U3, dim=0, p=float('inf'))))
    padding = (0, r - r_true, 0, r - r_true, 0, r - r_true)
    core_G_extended = torch.nn.functional.pad(core_G, padding)
    U1_extended = torch.cat((U1, torch.zeros(n, r - r_true)), dim=1)
    U2_extended = torch.cat((U2, torch.zeros(n, r - r_true)), dim=1)
    U3_extended = torch.cat((U3, torch.zeros(n, r - r_true)), dim=1)
    print(core_G_extended)
    return tl.tucker_to_tensor((core_G, [U1, U2, U3])), mu, core_G_extended, [U1_extended,U2_extended,U3_extended]


def matrixise(T, mode):
    return tl.unfold(T, mode)

def generate_random_mask(m, n, p, alpha):
    total_elements = m * n * p
    n_ones = int(alpha * total_elements)
    flat_mask = torch.zeros(total_elements, dtype=torch.int, device=device)
    flat_mask[:n_ones] = 1
    flat_mask = flat_mask[torch.randperm(total_elements, device=device)]
    return flat_mask.reshape(m, n, p)

def generate_uniform_random(a,b,size):
    return a + (b-a)*torch.rand(size, device=device)

def min_singular_value(matrix, eps=1e-5):
    singular_values = torch.linalg.svdvals(matrix)
    significant_values = singular_values[singular_values > eps]
    if len(significant_values) == 0:
        return 0
    return significant_values.min()



n_iter = 500
stepsize = 0.4

spectral_init = True
radius_init = 2
n = m = p = 20
r_true = 2
r = 6
kappa = 5

#Theorem 1
decay_constant = 1-0.45*stepsize #rho in paper

T_true, mu, G, factors = generate_tensor(n, r_true, r, kappa)
treshold_0 = 1.5*T_true.abs().max().item() #zeta_0 in paper
treshold_1 = (2*torch.sqrt(mu*r/n))**3*min( min_singular_value(matrixise(T_true, 0)),
                                        min_singular_value(matrixise(T_true, 1)),
                                        min_singular_value(matrixise(T_true, 2))) #zeta_1 in paper

corruption_factor = 100/torch.sqrt(kappa*((mu*r)**3)) #alpha in paper
corruption_factor = 0.2
corr_scaler = 1

T_true_corr = T_true + corr_scaler*torch.mul(generate_random_mask(n,n,n, corruption_factor),  generate_uniform_random(- 1* T_true.abs().mean().item(), T_true.abs().mean().item(), (n,n,n)))

errs = rpca(G, factors, T_true, T_true_corr, [r,r,r], treshold_0, treshold_1, stepsize, decay_constant, n_iter, 10**-10, device, spectral_init, perturb=radius_init)
errs_ours = rpca_ours(G, factors, T_true, T_true_corr, [r,r,r], treshold_0, n_iter, spectral_init, perturb=radius_init)


plt.figure(figsize=(8, 5))
plt.plot(torch.tensor(errs)/torch.norm(T_true), label='err_theirs')
plt.plot(torch.tensor(errs_ours)/torch.norm(T_true), label='err_ours')

# Add labels and title
plt.xlabel('iteration')
plt.ylabel('error')
plt.yscale('log')
plt.title(f'\| c(xt) - c(x*) \|_2 / \| c(x*) \|_2 , spectral init = {spectral_init}')
plt.legend()

# Show the plot
plt.show()