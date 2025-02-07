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

def c(X, Y, Z):
    """
    Computes the CP tensor from three factor matrices without using tensorly.

    Args:
        X (torch.Tensor): Factor matrix for mode-1 (shape: I x R)
        Y (torch.Tensor): Factor matrix for mode-2 (shape: J x R)
        Z (torch.Tensor): Factor matrix for mode-3 (shape: K x R)

    Returns:
        torch.Tensor: Reconstructed CP tensor of shape (I, J, K)
    """
    I, R1 = X.shape
    J, R2 = Y.shape
    K, R3 = Z.shape
    
    assert R1 == R2 == R3, "Factor matrices must have the same rank (R)"

    # Compute outer products and sum over rank R
    tensor = sum(
        X[:, r].view(I, 1, 1) * 
        Y[:, r].view(1, J, 1) * 
        Z[:, r].view(1, 1, K) 
        for r in range(R1)
    )

    return tensor

def nabla_c_transpose_g_sym(X, v):
    """
    Thank you GPT O 
    Compute (nabla c(X))^T g for given X and g.
    
    Parameters
    ----------
    X : torch.Tensor of shape (n, r)
        Factor matrix with r components, each of length n.
    v : torch.Tensor of shape (n, n, n)
        The 3D tensor 'g'.

    Returns
    -------
    torch.Tensor of shape (n, r)
        The result of (nabla c(X))^T g.
    """
    n, r = X.shape
                   
    # Let's use einsum directly:
    A = torch.einsum('ijk,il,jl->kl', v, X, X)  # sum over i,j
    # A: (k,l)

    B = torch.einsum('ijk,il,kl->jl', v, X, X)  # sum over i,k

    C = torch.einsum('ijk,jl,kl->il', v, X, X) 

    out = A + B + C  # (n,r)

    return out


def nabla_c_transpose_g_assym(X, Y,Z, v):
    """
    Thank you GPT O 
    Compute (nabla c(X))^T g for given X and g.
    
    Parameters
    ----------
    X : torch.Tensor of shape (m, r)
    Y:  torch.Tensor of shape (n, r)
    Z:  torch.Tensor of shape (p, r)
        Factor matrix with r components, each of length n.
    g : torch.Tensor of shape (n, n, n)
        The 3D tensor 'g'.

    Returns
    -------
    torch.Tensor of shape (m, r). (n,r) and (p,r)
        The result of (nabla c(X))^T g.
    """
    
    
    A = torch.einsum('ijk,jl,kl->il', v, Y, Z) # sum over i,j
    
    B = torch.einsum('ijk,il,kl->jl', v, X, Z)  # sum over i,k
 
    C = torch.einsum('ijk,il,jl->kl', v, X, Y)  # sum over j,k


    return A,B,C




def operator_sym(X, XPRIME):
    # Compute A = X^T X
    XX = X.T @ X
    XX_PRIME = XPRIME.T @ X
    
    
   

    # Compute the final result
    result = 3 * (XPRIME @ (XX*XX)) + 6 * (X @ (XX_PRIME * XX))
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
    r = g - (operator_sym(X, x) + damping * x)
    p = r.clone()

    # Compute initial residual norm squared
    rs_old = (r * r).sum()

    info = {
        'iterations': 0,
        'residual_norm': torch.sqrt(rs_old).item()
    }

    for i in range(max_iter):
        Ap = operator_sym(X, p) + damping * p
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



def operator_assym(X, Y, Z, XPRIME, YPRIME, ZPRIME):
    # Shape assumptions (for example):
    # X, XPRIME: (n, d), so X.T @ X -> (d, d), etc.
    
    XX = X.T @ X            # (d, d)
    YY = Y.T @ Y            # (d, d)
    ZZ = Z.T @ Z            # (d, d)

    XX_PRIME = XPRIME.T @ X # (d, d)
    YY_PRIME = YPRIME.T @ Y # (d, d)
    ZZ_PRIME = ZPRIME.T @ Z # (d, d)
    
    # -- Each RES_* is a sum of three terms, all shaped (n, d).
    # -- We keep a symmetrical pattern: 
    #    1) prime variable on the left, unprimed on the right
    #    2) unprimed variable on the left, "prime" factor in the middle, unprimed on the right
    #    etc.
    #
    # Notice that each term is: 
    #    <something of shape (n, d)> @ (<something of shape (d, d)> * <something else of shape (d, d)>)
    # i.e. standard matrix multiplication on the outside, but elementwise ("*") on the inside.

    RES_X = (
        XPRIME @ (YY * ZZ)            # prime-X times (unprimed Y & Z)
        + X @ (YY_PRIME * ZZ)         # unprimed X times (prime-Y, unprimed Z)
        + X @ (YY * ZZ_PRIME)         # unprimed X times (unprimed Y, prime-Z)
    )

    RES_Y = (
        YPRIME @ (XX * ZZ)            # prime-Y times (unprimed X & Z)
        + Y @ (XX_PRIME * ZZ)         # unprimed Y times (prime-X, unprimed Z)
        + Y @ (XX * ZZ_PRIME)         # unprimed Y times (unprimed X, prime-Z)
    )

    RES_Z = (
        ZPRIME @ (XX * YY)            # prime-Z times (unprimed X & Y)
        + Z @ (XX_PRIME * YY)         # unprimed Z times (prime-X, unprimed Y)
        + Z @ (XX * YY_PRIME)         # unprimed Z times (unprimed X, prime-Y)
    )

    return RES_X, RES_Y, RES_Z



def compute_preconditionner_applied_to_g_cp_assym(X, Y, Z, grad, damping, max_iter=100, epsilon=1e-14):
    """
    Thanks GPT O
    Conjugate gradient method. g is shape of X.
    X, g are PyTorch tensors.
    operator(X, v) should be defined to return a PyTorch tensor of the same shape as X.
    """
    # Initialize x as a zero tensor like g
    gx,gy,gz = torch.unbind(grad)
    
    x = torch.zeros_like(gx)
    y = torch.zeros_like(gy)
    z = torch.zeros_like(gz)

    # Compute initial residual
    r = torch.stack((gx,gy,gz)) - ( torch.stack(operator_assym(X,Y,Z,x,y,z)) + damping * torch.stack((x, y,z)))
    p = r.clone()

    # Compute initial residual norm squared
    rs_old = (r * r).sum()

    info = {
        'iterations': 0,
        'residual_norm': torch.sqrt(rs_old).item()
    }

    for i in range(max_iter):
        px,py,pz = torch.unbind(p)
        Ap = torch.stack(operator_assym(X,Y,Z,px,py,pz)) + damping * torch.stack((px,py,pz))
        pAp = (p * Ap).sum()
        alpha = rs_old / pAp

        x = x + alpha * px
        y = y + alpha * py
        z = z + alpha * pz
        
        r = r - alpha * Ap
        rs_new = (r * r).sum()

        info['iterations'] = i + 1
        info['residual_norm'] = torch.sqrt(rs_new).item()

        if rs_new.sqrt() <= epsilon:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x,y,z




def boot_strap_init(T_star,X_star, tol, n, r):
        
    
    err_rel = 0
    
    
    X = X_star.clone()
    pad_amount = r - X_star.shape[1]
    X = torch.nn.functional.pad(X, (0, pad_amount), mode='constant', value=0)
    

    
    to_add = 10**-5
    
    T = c(X,X,X)
    err_rel = torch.norm(T - T_star)/torch.norm(T_star)
            
    print(err_rel)

    while err_rel <= tol:
        X  += torch.rand(n,r)*to_add
        
        T = c(X,X,X) 
        err_rel = torch.norm(T - T_star)/torch.norm(T_star)
        print(err_rel)
        


    return X,X,X


def boot_strap_init_assym(T_star,X_star, Y_star, Z_star, tol, n,r):
    m = X_star.shape[0]
    n = Y_star.shape[0]
    p = Z_star.shape[0]
    
    
    X = X_star.clone()
    pad_amount = r - X_star.shape[1]
    X = torch.nn.functional.pad(X, (0, pad_amount), mode='constant', value=0)
    
        
    Y = Y_star.clone()
    pad_amount = r - Y_star.shape[1]
    Y = torch.nn.functional.pad(Y, (0, pad_amount), mode='constant', value=0)
        
    Z = Z_star.clone()
    pad_amount = r - Z_star.shape[1]
    Z = torch.nn.functional.pad(Z, (0, pad_amount), mode='constant', value=0)

    
    to_add = 10**-4
    
    T = c(X,Y,Z)
    err_rel = torch.norm(T - T_star)/torch.norm(T_star)


    while err_rel <= tol:
        print(err_rel)
        X  += torch.randn(m,r)*to_add
        Y  += torch.randn(n,r)*to_add
        Z  += torch.randn(p,r)*to_add
        
        T = c(X,Y,Z) 
        err_rel = torch.norm(T - T_star)/torch.norm(T_star)
        
        


    return X,Y,Z




def run_methods(methods_test, keys, n, r_true, target_d, identity, device, 
                n_iter, spectral_init, base_dir, loss_ord, radius_init, symmetric,
                corr_level=0, q=0.9, lambda_ = 0.0001, gamma = 0.001): 
    
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
        
        if symmetric:
            T_star = c(X_star,X_star, X_star)
        else:
            Q_uy, _ = torch.linalg.qr(torch.rand(n, r_true, device=device, dtype=torch.float64))
            Q_vy, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device, dtype=torch.float64))
            Q_uz, _ = torch.linalg.qr(torch.rand(n, r_true, device=device, dtype=torch.float64))
            Q_vz, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device, dtype=torch.float64))
            Y_star  = Q_uy @ S @ Q_vy.T
            Z_star = Q_uz @ S @ Q_vz.T
            T_star = c(X_star, Y_star, Z_star)
            
        
        y_true =  measurement_operator.A(T_star )
        num_ones = int(y_true.shape[0]*corr_level)
        mask_indices = np.random.choice(y_true.shape[0], size=num_ones, replace=False)
        mask = np.zeros(y_true.shape[0])
        mask[mask_indices] = 1 
        
        y_true = y_true + np.linalg.norm(y_true)*np.random.normal(size=y_true.shape[0])*mask
  
 
          
        if symmetric:
            X0, Y0,Z0 = boot_strap_init(T_star, X_star,radius_init, n,r)
        
        else:
            X0, Y0, Z0 = boot_strap_init_assym(T_star, X_star, Y_star, Z_star, radius_init, n,r)
        # Print the relative error

        for method in methods:
            X = X0.clone()
            Y = Y0.clone()
            Z = Z0.clone()
        
            
            errs = []
            
            for k in range(n_iter):
                
                
                if symmetric:
                    T = c(X,X,X)
                else:
                    T = c(X,Y,Z)
                    
                err = torch.norm(T - T_star)
                rel_err = err/(torch.norm(T_star))
                if torch.isnan(err) or rel_err > 1:
                    errs = errs +  [1 for _ in range(n_iter - len(errs)) ]
                    break
             
                    
                if k%20 == 0:
                    print(method)
                    print(k)
                    print(rel_err)  
                    print('---')
                errs.append(rel_err)
                
               
                residual = measurement_operator.A( T) - y_true
                
                if loss_ord == 1:
                    subgradient_h = measurement_operator.A_adj( torch.sign( residual ) ).view(-1) #L1
                    h_c_x =  torch.sum(torch.abs( residual )).item()
                elif loss_ord == 0.5:
                    subgradient_h = measurement_operator.A_adj( residual/torch.norm(residual) ).view(-1) #L2
                    h_c_x =  torch.norm(residual)
                elif loss_ord == 2:
                    subgradient_h = measurement_operator.A_adj( residual).view(-1) #L2
            
                    h_c_x =  0.5*torch.norm(residual)**2
                    
                    
                if symmetric:
                    grad = nabla_c_transpose_g_sym(X, subgradient_h.view(n,n,n))
                
                else:
                    grad = torch.stack(nabla_c_transpose_g_assym(X,Y,Z, subgradient_h.view(n,n,n)))
                
                if method in  ['Gradient descent', 'Subgradient descent']:
                    stepsize = h_c_x/(torch.norm(grad)**2)
                    preconditioned_grad = grad
                else:
                    damping = torch.sqrt(h_c_x) if loss_ord == 2 else h_c_x*10**-5
                    
                    damping = 0 if method == 'Gauss-Newton' else damping
                    #damping = 0 if method == 'Gauss-Newton' else (lambda_*q**k)
                    
                    if symmetric:
                        preconditioned_grad = compute_preconditionner_applied_to_g_cp_sym(X, grad, damping)
                    else:
                        preconditioned_grad = torch.stack(compute_preconditionner_applied_to_g_cp_assym(X, Y,Z, grad, damping) )
                    
        
                    stepsize = (h_c_x) / (torch.dot(subgradient_h,subgradient_h))
                
                #stepsize = gamma*q**(k) #geometric stepsize
                #stepsize = 0.1
                
                if symmetric:
                    X = X - stepsize * preconditioned_grad
                else:
                    prgx, prgy, prgz = torch.unbind(preconditioned_grad)
                    X = X - stepsize * prgx
                    Y = Y - stepsize * prgy
                    Z = Z - stepsize * prgz
            
            file_name = f'experiments/exptensor{"sym" if symmetric else ""}_{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_{0}.csv'
            full_path = os.path.join(base_dir, file_name)
            np.savetxt(full_path, np.array(errs), delimiter=',') 
            full_path = os.path.join(base_dir, file_name)
            outputs[method] = errs
            
    return outputs
                
        
        

n = 5
r_true = 2
target_d = n * r_true * 20
symmetric = False

identity = False 
device = 'cpu'
spectral_init = False
base_dir = os.path.dirname(os.path.abspath(__file__))
loss_ord = 1
radius_init = 10**-2  if symmetric else 10**-3
n_iter = 1000

np.random.seed(42)

keys = [(4,10)]

methods = [ 'Subgradient descent', 'Levenberg–Marquardt (ours)']

methods_test = methods

# Call the function
run_methods(methods_test, keys, n, r_true, target_d, identity, device, 
            n_iter, spectral_init, base_dir, 
            loss_ord, radius_init, symmetric)

errs, stds = collect_compute_mean(keys, loss_ord, r_true, False, methods, 'tensor' + ('sym' if symmetric else ''))
plot_losses_with_styles(errs, stds, r_true, loss_ord, base_dir, ('Symmetric' if symmetric else '') + 'CP', 1)
