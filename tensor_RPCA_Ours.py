#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:40:55 2024

@author: aglabassi
"""


import torch
import tensorly as tl
from itertools import product
from tensor_RPCA_Theirs import thre

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



def retrieve_tensors(concatenated_tensor, original_shapes):

   
    lengths = [torch.prod(torch.tensor(shape)).item() for shape in original_shapes]

    split_tensors = torch.split(concatenated_tensor, lengths)

    return [split_tensors[i].reshape(original_shapes[i]) for i in range(len(original_shapes))]


def random_perturbation(dimensions, delta):
    """
    Generates a random perturbation tensor with specified dimensions and a Frobenius norm of at least delta.

    Args:
    dimensions (tuple of ints): The shape of the tensor to generate.
    delta (float): The minimum Frobenius norm of the generated tensor.

    Returns:
    torch.Tensor: A tensor of the specified dimensions with a Frobenius norm of at least delta.
    """
    # Generate a random tensor with the specified dimensions
    perturbation = torch.randn(dimensions)

    # Calculate its current Frobenius norm
    current_norm = torch.norm(perturbation, p='fro')

    # Scale the tensor to have a Frobenius norm of exactly delta
    if current_norm > 0:
        scale_factor = delta / current_norm
        perturbation *= scale_factor
    
    # If current norm is exactly zero (rare), regenerate the tensor (recursive call)
    if current_norm == 0:
        return random_perturbation(dimensions, delta)

    return perturbation

def fill_tensor_elementwise(source, target):
    """Fill target tensor element-wise from source tensor."""
    if len(source.shape) == 1:
        for i in range(source.shape[0]):
            target[i] = source[i]
    elif len(source.shape) == 2:
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                target[i, j] = source[i, j]
    elif len(source.shape) == 3:
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                for k in range(source.shape[2]):
                    target[i, j, k] = source[i, j, k]


def rpca_ours(X,Y, ranks, z0, n_iter, G_true, factors_true, perturb=0.1):
    
    G0 = (G_true + random_perturbation(G_true.shape, perturb)).clone()
    factors0 = [ (f + random_perturbation(f.shape, perturb)).clone() for f in factors_true ]
    
    G = torch.zeros(*G0.shape)
    factors = [ torch.zeros(*f.shape) for f in factors0 ]
    fill_tensor_elementwise(G0, G)
    for idx, factor in enumerate(factors):
        fill_tensor_elementwise(factors0[idx], factor)

    errs = []
    shapes = [ t.shape for t in [G] + factors ]
    best_error = torch.sum(torch.abs(X - Y))
    
    for k in range(n_iter):
        residual = tl.tucker_to_tensor((G, factors)) - Y
        dist_to_sol_emb = torch.norm( tl.tucker_to_tensor((G, factors)) - X ).item()
        h_c_x =  torch.sum(torch.abs( residual )).item()
        
        print(dist_to_sol_emb)
        print(h_c_x)
        print(best_error)
        print('---')
        errs.append(torch.abs( h_c_x - best_error ).item())
        
        jac_c = compute_jacobian_c_tractable(G, *factors)
        
        concatenated = torch.cat([ t.reshape(-1)  for t in [G] + factors ])
        subgradient = torch.sign( residual ).reshape(-1)
        
        stepsize = (h_c_x - best_error) / (torch.dot(subgradient, subgradient))
        
        concatenated = concatenated - stepsize *  (torch.linalg.pinv( jac_c.T@jac_c + (dist_to_sol_emb)*torch.eye(jac_c.shape[1]) ) ) @ (jac_c.T @ subgradient )
        
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
    n,_ = U2.shape
    p,_ = U3.shape
    
    res = torch.zeros((m,r, m*n*p), dtype=torch.float32, device=G.device)
    eye_m = torch.eye(m, dtype=torch.float32, device=G.device)
    eye_r = torch.eye(r, dtype=torch.float32, device=G.device)
    for i_ in range(m):
        for a_ in range(r):
            E_ia = torch.outer(eye_m[i_], eye_r[a_])
            res[i_, a_] = (torch.kron(torch.kron(E_ia, U2), U3)@G.reshape(-1))
            
    return res.reshape(m*r, m*n*p).permute(1,0)


def jacobian_u2(G, U1, U2, U3):
    
    m, r = U1.shape
    n,_ = U2.shape
    p,_ = U3.shape
    
    res = torch.zeros((n, r, m*n*p), dtype=torch.float32, device=G.device)
    eye_n = torch.eye(n, dtype=torch.float32, device=G.device)
    eye_r = torch.eye(r, dtype=torch.float32, device=G.device)
    for i_ in range(n):
        for a_ in range(r):
            E_ia = torch.outer(eye_n[i_], eye_r[a_])
            res[i_, a_] = (torch.kron(torch.kron(U1, E_ia), U3)@G.reshape(-1))
            
    return res.reshape(n*r, m*n*p).permute(1,0)


def jacobian_u3(G, U1, U2, U3):
    
    m, r = U1.shape
    n,_ = U2.shape
    p,_ = U3.shape
    
    res = torch.zeros((p, r, m*n*p), dtype=torch.float32, device=G.device)
    eye_n = torch.eye(p, dtype=torch.float32, device=G.device)
    eye_r = torch.eye(r, dtype=torch.float32, device=G.device)
    for i_ in range(p):
        for a_ in range(r):
            E_ia = torch.outer(eye_n[i_], eye_r[a_])
            res[i_, a_] = (torch.kron(torch.kron(U1, U2), E_ia)@G.reshape(-1))
            
    return res.reshape(p*r, m*n*p).permute(1,0)

n=10
r=4
G = torch.rand(r, r, r, requires_grad=True)
A1 = torch.rand(n, r, requires_grad=True)
A2 = torch.rand(n, r, requires_grad=True)
A3 = torch.rand(n, r, requires_grad=True)

J1 = compute_jacobian_c_autograd(G, A1, A2, A3)
J2 = compute_jacobian_c_tractable(G,A1,A2,A3)

#assert torch.allclose(J1, J2, atol=1e-12), "The outputs are not equal!"


t1 = jacobian_g(G, A1, A2, A3).to(device)
t2 = jacobian_g_autograd(G, [A1, A2, A3]).to(device)
assert torch.allclose(t1, t2, atol=1e-12), "The outputs are not equal!"



t1 = jacobian_u1(G, A1, A2, A3).to(device)
t2 = jacobian_u_autograd(G, [A1, A2, A3], 0).to(device)
assert torch.allclose(t1, t2, atol=1e-12), "The outputs are not equal!"



t1 = jacobian_u2(G, A1, A2, A3).to(device)
t2 = jacobian_u_autograd(G, [A1, A2, A3], 1).to(device)
assert torch.allclose(t1, t2, atol=1e-12), "The outputs are not equal!"



t1 = jacobian_u3(G, A1, A2, A3).to(device)
t2 = jacobian_u_autograd(G, [A1, A2, A3], 2).to(device)
assert torch.allclose(t1, t2, atol=1e-12), "The outputs are not equal!"