#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:40:55 2024

@author: aglabassi
"""


import torch
import tensorly as tl
import numpy as np
import os
from utils import  fill_tensor_elementwise, retrieve_tensors, thre

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



def tensor_recovery(method, G_true, factors_true, G0_init, factors0_init, X,Y, measurement_operator, r_true, cond_number, ranks, z0, n_iter, spectral_init, base_dir, loss_ord, perturb=0.1, fix_G=True):
    r = ranks[0] 
    if spectral_init:
        G0, factors0 = tl.decomposition.tucker(Y - thre(Y, z0, device), rank=ranks)
    else:      
        G0 = G0_init.clone()
        factors0 = [ f.clone() for f in factors0_init ]
        
        
    r = G_true.shape[0]
    delim = r**3 if fix_G else 0
    
    G = torch.zeros(*G0.shape).to(device).double()
    factors = [ torch.zeros(*f.shape).to(device).double() for f in factors0 ]
    fill_tensor_elementwise(G0, G)
    for idx, factor in enumerate(factors):
        fill_tensor_elementwise(factors0[idx], factor)

    errs = []
    shapes = [ t.shape for t in [G] + factors ]
    best_error = torch.sum(torch.abs(X - Y)) if loss_ord == 1 else  torch.sum((X - Y)**2)
    
    for k in range(n_iter):
        
        Xk  =  tl.tucker_to_tensor((G, factors))
        residual = measurement_operator.A( Xk - Y )
        
        jac_c = compute_jacobian_c_tractable(G, *factors)
        
        concatenated = torch.cat([ t.reshape(-1)  for t in [G] + factors ])
        
        dist_to_sol_emb = torch.norm( Xk - X ).item()
        
        if loss_ord == 1:
            subgradient = measurement_operator.A_adj( torch.sign( residual ) ).reshape(-1) #L1
            h_c_x =  torch.sum(torch.abs( residual )).item()
        elif loss_ord == 2:
            subgradient = measurement_operator.A_adj( residual/torch.norm(residual) ).reshape(-1) #L2
            h_c_x =  torch.norm(residual)
        
        
        print(k)
        print(dist_to_sol_emb)
        if np.isnan(dist_to_sol_emb):
            errs = errs +  [1 for _ in range(n_iter - len(errs)) ]
            break
        print(h_c_x)
        print(best_error)      
        print('---')
        errs.append(dist_to_sol_emb /(torch.linalg.norm(X)))
        
        subgradient[:delim] = 0
        jac_c[:, :delim] = 0
        

        direction = 0
        damping = dist_to_sol_emb if method == 'Levenberg–Marquardt (ours)' else 0
        if method  == 'Gauss-Newton':
            try:
                direction = torch.linalg.lstsq( jac_c.T@jac_c,  jac_c.T @ subgradient).solution 
            except:
                direction = jac_c.T @ subgradient
                
        
        elif method == 'Levenberg–Marquardt (ours)':
            direction = torch.linalg.solve(jac_c.T@jac_c + (damping)*torch.eye(jac_c.shape[1]).to(device),jac_c.T @ subgradient )

        elif method in ['Subgradient descent', 'Gradient descent']:
            direction = jac_c.T @ subgradient
        else:
            raise NotImplementedError()
            
        
        if method == 'Gauss-Newton':
            stepsize = (h_c_x - best_error) / (torch.dot(direction, jac_c.T@jac_c@direction ))
            
        elif method == 'Levenberg–Marquardt (ours)':
            stepsize = (h_c_x - best_error) / (torch.dot(subgradient, subgradient))
        elif method in ['Subgradient descent', 'Gradient descent']:
            stepsize = (h_c_x - best_error) / (torch.dot(jac_c.T @ subgradient, jac_c.T @ subgradient))
            
        
        
        
        concatenated = concatenated - stepsize * direction
            
        tmp = retrieve_tensors(concatenated, shapes)

        G = tmp[0]
        factors = tmp[1:]
    
    file_name = f'experiments/exptensor_{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={cond_number}_trial_{0}.csv'
    full_path = os.path.join(base_dir, file_name)
    np.savetxt(full_path, np.array(errs), delimiter=',') 
    full_path = os.path.join(base_dir, file_name)
        
    return 'dont care'





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
    return torch.kron(torch.kron(U1, U2).double(), U3)




def jacobian_u1(G, U1, U2, U3):
    m, r = U1.shape 
    return torch.kron(torch.eye(m).to(device).double(), (torch.kron(U2, U3)) @ tl.unfold(G,0).T )
    


def jacobian_u2(G, U1, U2, U3):
    m, r = U1.shape
    n, _ = U2.shape
    p, _ = U3.shape
    return torch.kron(torch.eye(n).to(device).double(), (torch.kron(U1, U3)) @ tl.unfold(G,1).T ).view(n,m,p,-1).permute(1,0,2,3).reshape(m*n*p,-1)
    
def jacobian_u3(G, U1, U2, U3):
    m, r = U1.shape
    n, _ = U2.shape
    p, _ = U3.shape
    return torch.kron(torch.eye(p).to(device).double(), (torch.kron(U1, U2)) @ tl.unfold(G,2).T ).view(p,m,n,-1).permute(1,2,0,3).reshape(m*n*p,-1)

def generate_random_matrix_with_singular_values(n, r, L):
    """
    Generates a random matrix of size (n, r) with singular values defined by L.

    Args:
    - n (int): The number of rows of the matrix.
    - r (int): The number of columns of the matrix.
    - L (torch.Tensor): A tensor containing the desired singular values.

    Returns:
    - matrix (torch.Tensor): A random matrix of size (n, r) with controlled singular values.
    """
    # Ensure the number of singular values does not exceed the minimum dimension
    min_dim = min(n, r)
    if len(L) > min_dim:
        raise ValueError(f"The number of singular values cannot exceed {min_dim}")

    # Generate random orthogonal matrices U and V
    U, _ = torch.qr(torch.rand(n, n))
    V, _ = torch.qr(torch.rand(r, r))

    # Construct the singular value matrix
    S = torch.zeros(n, r)
    diag_indices = torch.arange(len(L))
    S[diag_indices, diag_indices] = L

    # Reconstruct the matrix using U, S, and V
    matrix = U @ S @ V.t()
    return matrix.requires_grad_(True)

def generate_random_matrix_with_singular_values(n, r, L):
    """
    Generates a random matrix of size (n, r) with singular values defined by L.

    Args:
    - n (int): The number of rows of the matrix.
    - r (int): The number of columns of the matrix.
    - L (torch.Tensor): A tensor containing the desired singular values.

    Returns:
    - matrix (torch.Tensor): A random matrix of size (n, r) with controlled singular values.
    """
    # Ensure the number of singular values does not exceed the minimum dimension
    min_dim = min(n, r)
    if len(L) > min_dim:
        raise ValueError(f"The number of singular values cannot exceed {min_dim}")

    # Generate random orthogonal matrices U and V
    U, _ = torch.qr(torch.rand(n, n))
    V, _ = torch.qr(torch.rand(r, r))

    # Construct the singular value matrix
    S = torch.zeros(n, r)
    diag_indices = torch.arange(len(L))
    S[diag_indices, diag_indices] = L

    # Reconstruct the matrix using U, S, and V
    matrix = U @ S @ V.t()

    return matrix.requires_grad_(True)




def jacobian_asymmetric_X(A1,A2):
    m, _ = A1.shape
    return torch.kron(torch.eye(m).to(device).double(), A2 )

def jacobian_asymmetric_Y(A1, A2):
    m, r = A1.shape
    n, _ = A2.shape
    return torch.kron(torch.eye(n).to(device).double(), A1 ).view(n,m,-1).permute(1,0,2).reshape(m*n,-1)
    
def compute_jacobian_assymetric(A1, A2):

    jacs = [jacobian_asymmetric_X(A1,A2), jacobian_asymmetric_Y(A1,A2) ]
    
    output = torch.cat(jacs, dim=1 )
    
    return output



    
n=3
r=2

L = torch.tensor([ 2.0 ])
# A2 = torch.rand(n,r).requires_grad_(True)
# A3 = torch.rand(n,r).requires_grad_(True)
# G = torch.rand(r,r,r).requires_grad_(True)

A1 = generate_random_matrix_with_singular_values(n, r, L)
A2 = generate_random_matrix_with_singular_values(n, r, L)
A3 = generate_random_matrix_with_singular_values(n, r, L)


G = torch.zeros(r,r,r)
G[torch.arange(r), torch.arange(r), torch.arange(r)] = 1
G = G.requires_grad_(True)

J1 = compute_jacobian_c_autograd(G, A1, A2, A3).double()
J2 = compute_jacobian_c_tractable(G,A1,A2,A3).double()

assert torch.allclose(J1, J2, atol=1e-12), "The outputs are not equal!"

print(len([eig for eig in sorted(torch.linalg.eigvals(J2[:,r**3:] @ J2[:,r**3:].T).real) if eig > 10**-6]))

jacobian_CP = J2[:,r**3:]








