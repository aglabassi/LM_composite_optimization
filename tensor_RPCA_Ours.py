#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:40:55 2024

@author: aglabassi
"""


import torch
import tensorly as tl
from utils import random_perturbation, fill_tensor_elementwise, retrieve_tensors, thre

tl.set_backend('pytorch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_jacobian_c_tractable(G, U1, U2, U3):

    jacs = [jacobian_g(G, U1,U2,U3), jacobian_u1(G, U1, U2, U3), jacobian_u2(G, U1, U2, U3) , jacobian_u3(G, U1, U2, U3) ]
    
    output = torch.cat(jacs, dim=1 )
    
    return output


def compute_jacobian_c_autograd(X, Y, Z, U):
    
    def c(G,A1,A2,A3):
        return tl.tucker_to_tensor((G, [A1,A2,A3]))
        #return torch.einsum('ia,jb,kc,abc->ijk', A1, A2, A3, G
    # Compute the output
    output = c(X, Y, Z, U)
    
    # Flatten the output to treat each element as a function output
    output_flat = output.reshape(-1)
    
    # Total number of outputs
    num_outputs = output_flat.numel()
    
    # Prepare the Jacobian matrix
    jacobian = []
    # Compute gradients for each output
    for i in range(num_outputs):
        # Zero gradients
        if X.grad is not None:
            X.grad.zero_()
        if Y.grad is not None:
            Y.grad.zero_()
        if Z.grad is not None:
            Z.grad.zero_()
        if U.grad is not None:
            U.grad.zero_()
        
        # Backward pass on the i-th output
        output_flat[i].backward(retain_graph=True)
        # Collect gradients
        jacobian.append(torch.cat([t.grad.flatten() for t in [X, Y, Z, U] if t.grad is not None]))

    # Stack to form the Jacobian matrix
    jacobian = torch.stack(jacobian, dim=0)
    
    return jacobian



def rpca_ours(G_true, factors_true, X,Y, ranks, z0, n_iter, spectral_init, perturb=0.1):
    
    if spectral_init:
        G0, factors0 = tl.decomposition.tucker(Y - thre(Y, z0, device), rank=ranks)
    else:      
        G0 = (G_true + random_perturbation(G_true.shape, perturb)).clone()
        factors0 = [ (f + random_perturbation(f.shape, perturb)).clone() for f in factors_true ]
        
        
    
    G = torch.zeros(*G0.shape).to(device)
    factors = [ torch.zeros(*f.shape).to(device) for f in factors0 ]
    fill_tensor_elementwise(G0, G)
    for idx, factor in enumerate(factors):
        fill_tensor_elementwise(factors0[idx], factor)

    errs = []
    shapes = [ t.shape for t in [G] + factors ]
    best_error = torch.sum(torch.abs(X - Y))
    
    for k in range(n_iter):
        
        Xk  =  tl.tucker_to_tensor((G, factors))
        residual = Xk - Y
        
        
        dist_to_sol_emb = torch.norm( Xk - X ).item()
        h_c_x =  torch.sum(torch.abs( residual )).item()
        
        print(k)
        print(dist_to_sol_emb)
        print(h_c_x)
        print(best_error)
        print('---')
        errs.append(dist_to_sol_emb)
        
        jac_c = compute_jacobian_c_tractable(G, *factors)
        
        concatenated = torch.cat([ t.reshape(-1)  for t in [G] + factors ])
        subgradient = torch.sign( residual ).reshape(-1).to(device)
        
        stepsize = (h_c_x - best_error) / (torch.dot(subgradient, subgradient))
        
        try:
            concatenated = concatenated - stepsize *  (torch.linalg.solve( jac_c.T@jac_c + (dist_to_sol_emb)*torch.eye(jac_c.shape[1]).to(device),  jac_c.T @ subgradient))
        except:
            concatenated = concatenated - stepsize * jac_c.T @ subgradient
        tmp = retrieve_tensors(concatenated, shapes)

        G = tmp[0]
        factors = tmp[1:]
        
        
    return errs





def jacobian_g_autograd(G, factors):
    _, r = factors[0].shape
    G.clone().detach().requires_grad_(True)
    factors = [ f.clone().detach().requires_grad_(True) for f in factors]
    return compute_jacobian_c_autograd(G, *factors)[:, :r**3]

    
def jacobian_u_autograd(G, factors, idx):
    n,r = factors[idx].shape
    G.clone().detach().requires_grad_(True)
    factors = [ f.clone().detach().requires_grad_(True) for f in factors]
    return compute_jacobian_c_autograd(G, *factors)[:, r**3 + idx*n*r: r**3 + (idx+1)*n*r]
    





def jacobian_g(G, U1,U2, U3):
    return torch.kron(torch.kron(U1, U2), U3)




def jacobian_u1(G, U1, U2, U3):
    m, r = U1.shape 
    return torch.kron(torch.eye(m).to(device), (torch.kron(U2, U3)) @ tl.unfold(G,0).T )
    


def jacobian_u2(G, U1, U2, U3):
    m, r = U1.shape
    n, _ = U2.shape
    p, _ = U3.shape
    return torch.kron(torch.eye(n).to(device), (torch.kron(U1, U3)) @ tl.unfold(G,1).T ).view(n,m,p,-1).permute(1,0,2,3).reshape(m*n*p,-1)
    
def jacobian_u3(G, U1, U2, U3):
    m, r = U1.shape
    n, _ = U2.shape
    p, _ = U3.shape
    return torch.kron(torch.eye(p).to(device), (torch.kron(U1, U2)) @ tl.unfold(G,2).T ).view(p,m,n,-1).permute(1,2,0,3).reshape(m*n*p,-1)
    
    
n=10
r=4



# Generate A1, A2, A3 with controlled singular values
A1 = torch.rand(n,r).requires_grad_(True)
A2 = torch.rand(n,r).requires_grad_(True)
A3 = torch.rand(n,r).requires_grad_(True)

G = torch.rand(r,r,r).requires_grad_(True)
J1 = compute_jacobian_c_autograd(G, A1, A2, A3)
J2 = compute_jacobian_c_tractable(G,A1,A2,A3)

assert torch.allclose(J1, J2, atol=1e-12), "The outputs are not equal!"

