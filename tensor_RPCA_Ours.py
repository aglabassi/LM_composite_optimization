#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:40:55 2024

@author: aglabassi
"""


import torch
import tensorly as tl
tl.set_backend('pytorch')

def tucker_product_optimized_torch(U1, U2, U3, G):
    return tl.tucker_to_tensor((G, [U1,U2,U3]))

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

def jacobian_u1_optimized(G, U1, U2, U3):
    n, r = U1.shape
    I_n = torch.eye(n, dtype=torch.float32, device=G.device).unsqueeze(2).unsqueeze(3)  # Shape: (n, n, 1, 1)
    I_r = torch.eye(r, dtype=torch.float32, device=G.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, r, r)

    outer_product = I_n * I_r  # Shape: (n, n, r, r)
    outer_product = outer_product.permute(2, 0, 1, 3)  # Shape: (r, n, n, n, n)

    outer_product_reshaped = outer_product.reshape(n * n * r, r)

    res = tucker_product_optimized_torch(outer_product_reshaped, U2, U3, G)

    res = res.reshape(r, n, n, n, n)

    return res

# Define dimensions
n = 10
r = 5

# Initialize random tensors
G = torch.rand(r, r, r, dtype=torch.float32)
U1 = torch.rand(n, r, dtype=torch.float32)
U2 = torch.rand(n, r, dtype=torch.float32)
U3 = torch.rand(n, r, dtype=torch.float32)

# Move tensors to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = G.to(device)
U1 = U1.to(device)
U2 = U2.to(device)
U3 = U3.to(device)

# Compute the Jacobians
t1 = jacobian_u1(G, U1, U2, U3).to(device)
t2 = jacobian_u1_optimized(G, U1, U2, U3).to(device)

print(t1)
print("----------")
print(t2)

# Assert statement to check equality
assert torch.allclose(t2, t1, atol=1e-6), "The outputs are not equal!"
