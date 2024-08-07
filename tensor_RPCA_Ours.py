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


def jacobian_c(X, Y, Z, U):
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

    return [split_tensors[i].view(original_shapes[i]).clone().detach().requires_grad_(True) for i in range(len(original_shapes))]


def rpca_ours(X,Y, ranks, z0, n_iter):
    
    G0, factors0 = tl.decomposition.tucker(Y - thre(Y, z0, device), rank=ranks)
    
    errs = []
    shapes = [ t.shape for t in [G0] + factors0 ]
    G = G0.clone().detach().requires_grad_(True)
    factors = [ factor.clone().detach().requires_grad_(True) for factor in factors0 ]
    best_error = torch.sum(torch.abs(X - Y))
    
    for k in range(n_iter):
        dist_to_sol_emb = torch.norm( tl.tucker_to_tensor((G, factors)) - Y  ).item()
        h_c_x =  torch.sum(torch.abs( tl.tucker_to_tensor((G, factors)) -  Y)).item()
        print(dist_to_sol_emb)
        print(h_c_x)
        print(best_error)
        print('---')
        errs.append(dist_to_sol_emb)
        
        jac_c = jacobian_c(G, *factors)
        
        concatenated = torch.cat([ t.reshape(-1)  for t in [G] + factors ])
        subgradient = torch.sign( tl.tucker_to_tensor((G, factors)) - Y  ).reshape(-1)
        
        stepsize = (h_c_x - best_error) / (torch.dot(subgradient, subgradient))
        
        concatenated = concatenated - stepsize *  (torch.linalg.pinv( jac_c.T@jac_c + 0*torch.eye(jac_c.shape[1]) ) ) @ (jac_c.T @ subgradient )
        
        tmp = retrieve_tensors(concatenated, shapes)

        G = tmp[0]
        factors = tmp[1:]
        
        
    return errs
    
    
