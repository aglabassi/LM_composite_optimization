# -*- coding: utf-8 -*-
#Author: Abdel
#Implementation of method in "Fast and Provable Tensor Robust Principal Component Analysis via Scaled Gradient Descent"
import numpy as  np
from tensor_RPCA_Theirs import rpca
import torch
import matplotlib.pyplot as plt

def generate_tensor(n,r, kappa):
    
    U1 = np.linalg.svd( np.random.rand(n,n))[0][:,:r]
    U2 = np.linalg.svd( np.random.rand(n,n))[0][:,:r]
    U3 = np.linalg.svd( np.random.rand(n,n))[0][:,:r]
    core_G = np.zeros((r, r, r))
    indices = np.arange(r)
    core_G[indices, indices, indices] = kappa ** (-indices / (r - 1))
    
    mu  = (n/r)*max(np.max(np.linalg.norm(U1, ord=np.inf, axis=0)),
                    np.max(np.linalg.norm(U2, ord=np.inf, axis=0)),
                    np.max(np.linalg.norm(U3, ord=np.inf, axis=0)))
    
    return tucker_product_optimized(U1,U2,U3,core_G), mu


def soft_shrink(T, treshold):
    return np.multiply( np.sign(T), np.maximum(0, np.abs(T) - treshold))

def tucker_product(U1, U2, U3, G):
    # Get dimensions
    I, R1 = U1.shape
    J, R2 = U2.shape
    K, R3 = U3.shape
    assert G.shape == (R1, R2, R3), "Core tensor dimensions must match the second dimensions of factor matrices."

    result = np.zeros((I, J, K))

    # Calculate Tucker product
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for a in range(R1):
                    for b in range(R2):
                        for c in range(R3):
                            result[i, j, k] += U1[i, a] * U2[j, b] * U3[k, c] * G[a, b, c]

    return result

def tucker_product_optimized(U1,U2,U3,G):
    return np.einsum('ia,jb,kc,abc->ijk', U1, U2, U3, G)

def matrixise(T, mode):
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def hosvd(T,r):
    
    ndims = T.ndim
    Us = []
    
    for mode in range(ndims):
        unfolded = matrixise(T, mode)
        U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        U = U[:, :r]
        Us.append(U)
        
    core_matrix = tucker_product_optimized(Us[0].T, Us[1].T, Us[2].T, T)
        
        
    return tuple(Us + [ core_matrix ])
    

def update_S(U1, U2, U3, G, T_true_corr, k, decay_constant, treshold):
    treshold_k = treshold*decay_constant**k
    return soft_shrink(T_true_corr - tucker_product_optimized(U1,U2,U3,G), treshold_k)
    
    

def update_factors(S, T_true_corr, U1, U2, U3, G, stepsize):
    U1_hat = np.kron( U3, U2 ) @ matrixise(G, 0 ).T 
    U2_hat = np.kron( U3, U1 ) @ matrixise(G, 1 ).T 
    U3_hat = np.kron( U2, U1 ) @ matrixise(G, 2 ).T 
    
    U1_new = (1-stepsize)*U1 +  stepsize*((matrixise(S, 0) - matrixise(T_true_corr, 0)) @ U1_hat @ np.linalg.inv(U1_hat.T @ U1_hat)  )
    U2_new = (1-stepsize)*U2 +  stepsize*((matrixise(S, 1) - matrixise(T_true_corr, 1)) @ U2_hat @ np.linalg.inv(U2_hat.T @ U2_hat)  )
    U3_new = (1-stepsize)*U3 +  stepsize*((matrixise(S, 2) - matrixise(T_true_corr, 2)) @ U3_hat @ np.linalg.inv(U3_hat.T @ U3_hat)  )
    
    U1_tilde = np.linalg.inv(U1.T @ U1) @ U1.T
    U2_tilde = np.linalg.inv(U2.T @ U2) @ U2.T
    U3_tilde = np.linalg.inv(U3.T @ U3) @ U3.T
    G_new = (1-stepsize)*G  - stepsize* tucker_product_optimized(U1_tilde, U2_tilde, U3_tilde, S - T_true_corr)
    
    return U1_new, U2_new, U3_new, G_new


def generate_random_mask(m, n, p, alpha):
    
    total_elements = m * n * p 
    n_ones = int(alpha * total_elements)
    
    flat_mask = np.zeros(total_elements, dtype=int)
    flat_mask[:n_ones] = 1
    
    np.random.shuffle(flat_mask)
    
    mask = flat_mask.reshape(m, n, p)
    
    return mask


def min_singular_value(matrix, eps=1e-10):

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    nonzero_singular_values = singular_values[singular_values > eps]
    
    if len(nonzero_singular_values) == 0:
        return 0
    
    return np.min(nonzero_singular_values)


    
    
    
n_iter = 1000
stepsize = 0.5


n = m = p = 100
r = r_true = 20
kappa = 10

#Theorem 1
decay_constant = 1-0.45*stepsize #rho in paper

T_true, mu = generate_tensor(n, r_true, kappa)
treshold_0 = 1.5*np.max(np.abs(T_true)) #zeta_0 in paper
treshold_1 = (2*np.sqrt(mu*r/n))**3*min( min_singular_value(matrixise(T_true, 0)),
                                        min_singular_value(matrixise(T_true, 1)),
                                        min_singular_value(matrixise(T_true, 2))) #zeta_1 in paper

corruption_factor = 100/np.sqrt(kappa*((mu*r)**3)) #alpha in paper

T_true_corr = T_true + np.multiply(generate_random_mask(n,n,n, corruption_factor), np.random.uniform( - 1* np.mean(np.abs(T_true)), np.mean(np.abs(T_true)), size=(n,n,n)))

X = torch.FloatTensor(T_true)
Y = torch.FloatTensor(T_true_corr)
torch.set_printoptions(precision=10)

errs,_ = rpca(X, Y, [r,r,r], treshold_0, treshold_1, stepsize, decay_constant, n_iter, 0, 'cpu')

plt.plot(errs)
plt.yscale('log')




def jacobian_u1(G,U1,U2,U3):
    n,r = U1.shape
    res = np.zeros((r, n, n, n, n))
    for i_ in range(n):#TODO VECTORIZE THIS LOOPS
        for a_ in range(r):
            res[a_,i_] =  tucker_product_optimized(np.outer(np.eye(n, dtype=int)[i_], np.eye(r, dtype=int)[a_]), U2,U3,G)         
    return res
    
def jacobian_u1_optimized(G, U1, U2, U3):
    n, r = U1.shape
    I_n = np.eye(n)[:, :, np.newaxis, np.newaxis]  # Shape: (n, n, 1, 1)
    I_r = np.eye(r)[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, r, r)
    
    outer_product = I_n * I_r  # Shape: (n, n, r, r)
    
    outer_product_reshaped = outer_product.reshape(n * n * r, r)
    
    res = tucker_product_optimized(outer_product_reshaped, U2, U3, G)
    
    res = res.reshape(r, n, n, n, n)
    
    return res

n=5
r=1
G = np.random.rand(r,r,r)
U1 = np.random.rand(n,r)
U2 = np.random.rand(n,r)
U3 = np.random.rand(n,r)

print(jacobian_u1(G, U1, U2, U3))
print('------------')
print(jacobian_u1_optimized(G, U1, U2, U3))
assert(np.linalg.norm(jacobian_u1_optimized(G, U1, U2, U3) - jacobian_u1(G, U1, U2, U3)) == 0) #currently works only for r=1