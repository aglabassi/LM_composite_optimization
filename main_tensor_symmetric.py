# -*- coding: utf-8 -*-
# Author: Abdel 

import torch
import tensorly as tl
from tensorly.decomposition import symmetric_parafac_power_iteration
import os
from utils import collect_compute_mean, plot_losses_with_styles

tl.set_backend('pytorch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

# Ensure double precision globally where possible
torch.set_default_dtype(torch.float64)

class TensorMeasurementOperator:
    def __init__(self, n1, n2, n3, m, identity=False):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.m = m
        self.identity = identity
        
        # Generate m measurement tensors Ai ~ N(0, 1/m) in double precision
        self.A_tensors = torch.randn(m, n1, n2, n3, device=device, dtype=torch.float64) / torch.sqrt(torch.tensor(m, dtype=torch.float64, device=device))

    def A(self, X):
        if self.identity:
            # Flatten X and return
            return X.flatten()
        else:
            # Apply operator A(X⋆) = {⟨Ai, X ⟩}_m
            # einstein sum: i for A index, jkl for A dimension, jkl for X dimension
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

def c(X, weights= None):
    # c(X) = sum_{l=1}^{r} X_{:,l} ⊗ X_{:,l} ⊗ X_{:,l}
    # We can construct it using a Tucker decomposition with a superdiagonal core.
    r = X.shape[1]
    G  = torch.zeros(r, r, r, device=device, dtype=torch.float64)
    G[torch.arange(r), torch.arange(r), torch.arange(r)] = 1
    return tl.tucker_to_tensor((G, [X, X, X]))


def nabla_c_transpose_g(X, g):
    """
    Thank you GPT O 
    Compute (nabla c(X))^T g for given X and g.
    
    Parameters
    ----------
    X : torch.Tensor of shape (n, r)
        Factor matrix with r components, each of length n.
    g : torch.Tensor of shape (n, n, n)
        The 3D tensor 'g'.

    Returns
    -------
    torch.Tensor of shape (n, r)
        The result of (nabla c(X))^T g.
    """
    n, r = X.shape
    
    # A_l(k) = sum_{i,j} g[i,j,k] * X[i,l]*X[j,l]
    # We can compute this for all l,k by tensordot twice:
    # Step 1: contract over i: sum_{i} g[i,j,k]*X[i,l]
    # g is (n,n,n), X is (n,r)
    # torch.tensordot(g,X,(0,0)) results in shape (n,n,r) summing over i.
    tmpA = torch.tensordot(g, X, dims=([0],[0]))  # (n, n, r)
    # Now sum over j with X again: tmpA is (n,n,r), X is (n,r)
    # We want sum_j tmpA[j,k,l]*X[j,l], that is a bilinear form producing (n,r):
    # tensordot(tmpA, X, ([0],[0])) gives (n,r,r)
    # We want the diagonal over the last two r-dim, i.e. same l for both multiplications:
    # But we used X in both contractions in the same manner, resulting in a "matrix" over l,l'.
    # Instead, swap order: do the contraction with X on one mode fully first:
    
    # Let's do a more straightforward approach using einsum for clarity:

    # A_l:
    # A_l(k) = sum_{i,j} g[i,j,k]*X[i,l]*X[j,l]
    # We can write this as:
    # A(k,l) = (X.T g[:,:,k] X)_(l,l) but we just need the diagonal in l,l
    # Let's use einsum directly:
    A = torch.einsum('ijk,il,jl->kl', g, X, X)  # sum over i,j
    # A: (k,l)

    # B_l:
    # B_l(j) = sum_{i,k} g[i,j,k]*X[i,l]*X[k,l]
    # Just permute indices to get a similar form:
    # This is like A but with indices permuted: (i,j,k) -> (i,k,j)
    # We can get this by transposing g or by changing the einsum pattern:
    B = torch.einsum('ijk,il,kl->jl', g, X, X)  # sum over i,k
    # B: (j,l)

    # C_l:
    # C_l(i) = sum_{j,k} g[i,j,k]*X[j,l]*X[k,l]
    # Another permutation:
    C = torch.einsum('ijk,jl,kl->il', g, X, X)  # sum over j,k
    # C: (i,l)

    # Now we have A(k,l), B(j,l), C(i,l).
    # i,j,k all run from 0 to n-1, so we can just rename indices consistently:
    # The dimension indexes i,j,k are all size n, so we can add them directly as they are all vectors of length n for each l.
    # But A(k,l), B(j,l), and C(i,l) are indexed by different letters. We must align them.
    # Actually, i,j,k are dummy indices and are identical in range. We can just treat them as the same dimension and add them up.
    # Just rename k->i and j->i so that A, B, C all index with i. This is just a conceptual rename:
    # We'll do it by expanding each into shape (n,r) and add:

    # A, B, C are each (n,r), just with different dimension names. We'll just add them:
    out = A + B + C  # (n,r)

    return out


def operator(X, XPRIME):
    # Compute A = X^T X
    A = X.T @ X
    
    # Compute A squared elementwise
    A_sq = A * A  # or A**2

    # Compute the elementwise product (X'^T X) * (X^T X)
    B = (XPRIME.T @ X) * A

    # Compute the final result
    result = 3 * (XPRIME @ A_sq) + 6 * (X @ B)
    return result


def compute_preconditionner_applied_to_g_cp_sym(X, g, damping, max_iter=100, epsilon=1e-13):
    """
    Thanks GPT O
    Conjugate gradient method. g is shape of X.
    X, g are PyTorch tensors.
    operator(X, v) should be defined to return a PyTorch tensor of the same shape as X.
    """
    # Initialize x as a zero tensor like g
    x = torch.zeros_like(g)

    # Compute initial residual
    r = g - (operator(X, x) + damping * x)
    p = r.clone()

    # Compute initial residual norm squared
    rs_old = (r * r).sum()

    info = {
        'iterations': 0,
        'residual_norm': torch.sqrt(rs_old).item()
    }

    for i in range(max_iter):
        Ap = operator(X, p) + damping * p
        pAp = (p * Ap).sum()
        alpha = rs_old / pAp

        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()

        info['iterations'] = i + 1
        info['residual_norm'] = torch.sqrt(rs_new).item()

        if rs_new.sqrt() <= epsilon:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def boot_strap_init(T_star, tol, n, r):
        

    X  = torch.rand(n,r)
    
    measurement_operator = TensorMeasurementOperator(n, n, n, target_d, identity=True)
    idx = 0
    for k in range(1000):
       
        
        T  =  c(X)
        err = torch.norm(T - T_star)
        err_rel = err/ torch.norm(T_star)
        print(err_rel)
       
        
        if err_rel <= tol:
            break 
        
        residual = measurement_operator.A( T - T_star )
        
        
        

        subgradient_h = measurement_operator.A_adj( residual/torch.norm(residual) ).reshape(-1) #L2
        h_c_x =  torch.norm(residual)
        

        grad = nabla_c_transpose_g(X, subgradient_h.view(n,n,n))

        damping = err
        preconditioned_grad = compute_preconditionner_applied_to_g_cp_sym(X, grad, damping)
        
        stepsize = (h_c_x) / (torch.dot(subgradient_h,subgradient_h))
       
        X = X - stepsize * preconditioned_grad
        idx+=1
        
    print('err_rel' + str(err_rel) + ', ' + str(idx))
    
    while err_rel <= tol/10:
        X  += torch.rand(n,r)*0.0000001
        T = c(X)
        err_rel = torch.norm(T - T_star)/torch.norm(T_star)
        print(err_rel)

    return X



def run_methods(methods_test, keys, n, r_true, target_d, identity, device, n_iter, spectral_init, base_dir, loss_ord, radius_init, corr_level=0, q=0.9, lambda_ = 0.0001, gamma = 0.001): 
    
    measurement_operator = TensorMeasurementOperator(n, n, n, target_d, identity=identity)

    for key in keys:
        outputs = dict()    
        r, kappa = key
        
        Q_u, _ = torch.linalg.qr(torch.rand(n, r_true, device=device, dtype=torch.float64))
        Q_v, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device, dtype=torch.float64))
        
        # Create singular values as double
        singular_values = torch.linspace(1.0, 1/kappa, r_true, device=device, dtype=torch.float64)
        S = torch.diag(singular_values)
        
        # Construct X_star in double
        X_star = Q_u @ S @ Q_v.T
        T_star = c(X_star)
        
        y_true =  measurement_operator.A(T_star )
        num_ones = int(y_true.shape[0]*corr_level)
        mask_indices = np.random.choice(y_true.shape[0], size=num_ones, replace=False)
        mask = np.zeros(y_true.shape[0])
        mask[mask_indices] = 1 
        
        y_true = y_true + np.linalg.norm(y_true)*np.random.normal(size=y_true.shape[0])*mask
  
 

      
        X0 = boot_strap_init(T_star, radius_init, n,r)
        # Print the relative error
        
        for method in methods:
            X = X0.clone()
          
            
            errs = []
            
            for k in range(n_iter):
                T  =  c(X)
                err = torch.norm(T - T_star)
                rel_err = err/(torch.norm(T_star))
                if torch.isnan(err) or rel_err > 1:
                    errs = errs +  [1 for _ in range(n_iter - len(errs)) ]
                    break
                if rel_err < 10**-13:
                    errs = errs +  [10**-13 for _ in range(n_iter - len(errs)) ]
                    break
                    
                print(err)      
                print('---')
                errs.append(rel_err)
                
               
                residual = measurement_operator.A( T) - y_true
                
                if loss_ord == 1:
                    subgradient_h = measurement_operator.A_adj( torch.sign( residual ) ).reshape(-1) #L1
                    h_c_x =  torch.sum(torch.abs( residual )).item()
                elif loss_ord == 2:
                    subgradient_h = measurement_operator.A_adj( residual/torch.norm(residual) ).reshape(-1) #L2
                    h_c_x =  torch.norm(residual)
                

                grad = nabla_c_transpose_g(X, subgradient_h.view(n,n,n))

                if method in  ['Gradient descent', 'Subgradient descent']:
                    stepsize = h_c_x/(torch.norm(grad)**2)
                    preconditioned_grad = grad
                else:
                    damping = 0 if method == 'Gauss-Newton' else (lambda_*q**k)
                    preconditioned_grad = compute_preconditionner_applied_to_g_cp_sym(X, grad, damping)
                    
                    if method == 'Gauss-Newton':
                        QTT1 = preconditioned_grad
                        QTT2 = operator(X,preconditioned_grad)
                        denom = torch.sum( QTT1 * QTT2 )
                        stepsize = (h_c_x) / denom
                    elif method == 'Levenberg–Marquardt (ours)':
                        stepsize = (h_c_x) / (torch.dot(subgradient_h,subgradient_h))
                
                stepsize = gamma*q**(k) #geometric stepsize
                X = X - stepsize * preconditioned_grad
                    
                    
                    
                    
                    
                
            file_name = f'experiments/exptensorsym_{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_{0}.csv'
            full_path = os.path.join(base_dir, file_name)
            np.savetxt(full_path, np.array(errs), delimiter=',') 
            full_path = os.path.join(base_dir, file_name)
            outputs[method] = errs
    return outputs
                
        
        

n = 20
r_true = 2
target_d = n * r_true * 20
identity = False
device = 'cpu'
spectral_init = False
base_dir = os.path.dirname(os.path.abspath(__file__))
loss_ord = 2
radius_init = 0.00001
n_iter = 1000

keys = [(2,1), (2,10), (4,1), (4,10)]
if loss_ord == 1:
    methods = ['Subgradient descent', 'Gauss-Newton','Levenberg–Marquardt (ours)']
elif loss_ord == 2:
    methods = ['Gradient descent',  'Gauss-Newton', 'Levenberg–Marquardt (ours)']

methods_test = methods

# Call the function
run_methods(methods_test, keys, n, r_true, target_d, identity, device, 
            n_iter, spectral_init, base_dir, 
            loss_ord, radius_init)

errs, stds = collect_compute_mean(keys, loss_ord, r_true, False, methods, 'tensorsym')
plot_losses_with_styles(errs, stds, r_true, loss_ord, base_dir, 'Symmetric CP', 1)
