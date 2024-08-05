#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:40:55 2024

@author: aglabassi
"""


import torch
import tensorly as tl
from itertools import product

tl.set_backend('pytorch')

def tucker_product_optimized_torch(U1, U2, U3, G):
    
    return tl.tucker_to_tensor((G, [U1,U2,U3]))


def outer_product(vectors):
    
    dims = len(vectors)
    einsum_str = ','.join(chr(97 + i) for i in range(dims)) + '->' + ''.join(chr(97 + i) for i in range(dims))
    return torch.einsum(einsum_str, *vectors)

def jacobian_u1(G, U1, U2, U3):
    
    n, r = U1.shape
    res = torch.zeros((r, n, n, n, n), dtype=torch.float32, device=G.device)
    for i_ in range(n):
        for a_ in range(r):
            eye_n = torch.eye(n, dtype=torch.float32, device=G.device)
            eye_r = torch.eye(r, dtype=torch.float32, device=G.device)
            res[a_, i_] = tucker_product_optimized_torch(
                torch.outer(eye_n[i_], eye_r[a_]), U2, U3, G
            )
    return res

def jacobian_u2(G, U1, U2, U3):
    
    n, r = U2.shape
    res = torch.zeros((r, n, n, n, n), dtype=torch.float32, device=G.device)
    for i_ in range(n):
        for a_ in range(r):
            eye_n = torch.eye(n, dtype=torch.float32, device=G.device)
            eye_r = torch.eye(r, dtype=torch.float32, device=G.device)
            res[a_, i_] = tucker_product_optimized_torch(
                torch.outer(eye_n[i_], eye_r[a_]), U1, U3, G
            )
    return res


def jacobian_u_optimized(G, factors, idx):
    
    n, r = factors[idx].shape
    
    I_n = torch.eye(n, dtype=torch.float32, device=G.device).unsqueeze(2).unsqueeze(3)  # Shape: (n, n, 1, 1)
    I_r = torch.eye(r, dtype=torch.float32, device=G.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, r, r)

    outer_product = I_n * I_r  # Shape: (n, n, r, r)
    outer_product = outer_product.permute(2, 0, 1, 3)  # Shape: (r, n, n, n, r)

    outer_product_reshaped = outer_product.reshape(n * n * r, r)

    res = tl.tucker_to_tensor((G, [outer_product_reshaped] + factors[:idx] + factors[idx+1:]))

    res = res.reshape(r, n, n, n, n)

    return res


def jacobian_g(G, factors):
    
    rs = list(G.shape)
    rs_range = [ range(0,r) for r in rs]
    ns = [ f.shape[0] for f in factors ]
    res = torch.zeros(*rs + ns)
    
    for item in product(*rs_range):
        res[item] = outer_product(list(factor[:,r] for (factor,r) in zip(factors, item)))
    
    return res #dimension
    

def jacobian_g_optimized(G, factors):

    return torch.einsum('ia,jb,kc->abcijk', *factors)


def jacobian_c(G, factors):
    
    tensors = [jacobian_g_optimized(G, factors)] + [ jacobian_u_optimized(G, factors, idx) for idx in range(len(factors)) ]
    output_shape = [ f.shape[0] for f in factors ]
    output = torch.cat([ tensor.view(-1, *output_shape) for tensor in tensors], dim=0 )
    
    return output.view(output.shape[0], -1).transpose(1,0)
    

n = 100
r = 2


G = torch.rand(r, r, r, dtype=torch.float32)
U1 = torch.rand(n, r, dtype=torch.float32)
U2 = torch.rand(n, r, dtype=torch.float32)
U3 = torch.rand(n, r, dtype=torch.float32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = G.to(device)
U1 = U1.to(device)
U2 = U2.to(device)
U3 = U3.to(device)

t1 = jacobian_u2(G, U1, U2, U3).to(device)
t2 = jacobian_u_optimized(G, [U1, U2, U3], 1).to(device)

print(t1)
print("----------")
print(t2)

assert torch.allclose(t2, t1, atol=1e-12), "The outputs are not equal!"


t1 = jacobian_g(G, [U1, U2, U3]).to(device)
t2 = jacobian_g_optimized(G, [U1, U2, U3]).to(device)

print(t1)
print("----------")
print(t2)

assert torch.allclose(t2, t1, atol=1e-12), "The outputs are not equal!"

A = jacobian_c(G, [U1, U2, U3])
print(A.shape)
print(torch.linalg.pinv(A.T@A).shape)