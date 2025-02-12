#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Abdel Ghani Labassi
Date: February 12th 2025

This code implements a unified scheme that works for both CP–tensor and 
matrix factorization models. For CP tensor factorization we have
    T = CP(X,Y,Z) = einsum('ir,jr,kr->ijk', X, Y, Z),
while for matrix factorization one may consider
    T = XX^T   (symmetric)  or   T = XY^T   (asymmetric).
A Boolean flag "tensor" is used to switch between these models.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import collect_compute_mean, plot_losses_with_styles

# Ensure double precision globally where possible.
torch.set_default_dtype(torch.float64)

# Set device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


###############################################################################
# 1. CP Tensor Reconstruction
###############################################################################
def cp_reconstruct(X, Y, Z):
    """
    Reconstructs the CP tensor from three factor matrices using einsum.
    
    Args:
        X (torch.Tensor): Factor matrix for mode 1 of shape (I, R).
        Y (torch.Tensor): Factor matrix for mode 2 of shape (J, R).
        Z (torch.Tensor): Factor matrix for mode 3 of shape (K, R).
    
    Returns:
        torch.Tensor: A tensor of shape (I, J, K).
    """
    return torch.einsum('ir,jr,kr->ijk', X, Y, Z)


###############################################################################
# 2. Unified Local Initialization
###############################################################################
def unified_local_init(T_star, factors, dims, tol, r, symmetric, tensor=True):
    """
    Performs local initialization for both CP–tensor and matrix factorization.
    
    Args:
        T_star (torch.Tensor): The target tensor (or matrix).
        factors (list[torch.Tensor]): 
            - If tensor==True:
                • symmetric: [X_star] 
                • asymmetric: [X_star, Y_star, Z_star]
            - If tensor==False:
                • symmetric: [X_star]
                • asymmetric: [X_star, Y_star]
        dims (list[int]): The row–dimensions of each factor.
        tol (float): Tolerance for the relative error.
        r (int): Target number of columns (rank) after padding.
        symmetric (bool): Whether the model is symmetric.
        tensor (bool): If True, use CP–tensor reconstruction; else use matrix factorization.
    
    Returns:
        list[torch.Tensor]: The updated factor matrices.
    """
    new_factors = []
    for fac in factors:
        pad_amount = r - fac.shape[1]
        new_factors.append(F.pad(fac, (0, pad_amount), mode='constant', value=0))
    
    if tensor:
        if symmetric:
            T = cp_reconstruct(new_factors[0], new_factors[0], new_factors[0])
        else:
            T = cp_reconstruct(new_factors[0], new_factors[1], new_factors[2])
    else:
        if symmetric:
            T = new_factors[0] @ new_factors[0].T
        else:
            T = new_factors[0] @ new_factors[1].T
    
    err_rel = torch.norm(T - T_star) / torch.norm(T_star)
    to_add = 1e-5 
    while err_rel <= tol:
        if tensor:
            if symmetric:
                new_factors[0] = new_factors[0] + torch.rand(dims[0], r, device=new_factors[0].device) * to_add
            else:
                new_factors[0] = new_factors[0] + torch.randn(dims[0], r, device=new_factors[0].device) * to_add
                new_factors[1] = new_factors[1] + torch.randn(dims[1], r, device=new_factors[1].device) * to_add
                new_factors[2] = new_factors[2] + torch.randn(dims[2], r, device=new_factors[2].device) * to_add
        else:
            if symmetric:
                new_factors[0] = new_factors[0] + torch.rand(dims[0], r, device=new_factors[0].device) * to_add
            else:
                new_factors[0] = new_factors[0] + torch.randn(dims[0], r, device=new_factors[0].device) * to_add
                new_factors[1] = new_factors[1] + torch.randn(dims[1], r, device=new_factors[1].device) * to_add
        
        if tensor:
            if symmetric:
                T = cp_reconstruct(new_factors[0], new_factors[0], new_factors[0])
            else:
                T = cp_reconstruct(new_factors[0], new_factors[1], new_factors[2])
        else:
            if symmetric:
                T = new_factors[0] @ new_factors[0].T
            else:
                T = new_factors[0] @ new_factors[1].T
        
        err_rel = torch.norm(T - T_star) / torch.norm(T_star)
    return new_factors


###############################################################################
# 3. Helper: Split a Flattened Tensor into Blocks
###############################################################################
def split(concatenated, shapes):
    """
    Splits a flattened tensor into blocks with specified shapes.
    
    Args:
        concatenated (torch.Tensor): A 1D tensor.
        shapes (list[tuple]): List of shapes, e.g. [(m, r), (n, r), ...].
    
    Returns:
        tuple[torch.Tensor]: The split tensors.
    """
    output = []
    start = 0
    for shape in shapes:
        numel = np.prod(shape)
        block = concatenated[start:start + numel].reshape(shape)
        output.append(block)
        start += numel
    return tuple(output)


###############################################################################
# 4. Unified Linear Operator Function for CG
###############################################################################
def unified_operator(update, factors, symmetric, tensor=True):
    """
    Computes the action of the linearized operator (nabla c * nabla c^T)
    on a search direction “update”. This is used inside the conjugate–gradient solver.
    
    For the CP–tensor model (tensor=True):
      - Symmetric: factors = [X] and
          A(dX) = 3*(dX @ (XX ∘ XX)) + 6*(X @ ((dX.T @ X) ∘ XX))
      - Asymmetric: factors = [X, Y, Z] and
          A(dX, dY, dZ) is computed block–wise.
    
    For matrix factorization (tensor=False):
      - Symmetric: factors = [X] and
          A(g) = (2*g_mat @ (X.T @ X) + 2*X @ (g_mat.T @ X)).reshape(-1)
      - Asymmetric: factors = [X, Y] and
          A(g_x, g_y) is computed as the concatenation of
              (X @ (g_y.T @ Y) + g_x @ (Y.T @ Y)) and
              (Y @ (g_x.T @ X) + g_y @ (X.T @ X)).
    
    Args:
        update (torch.Tensor): The update direction (flattened or not).
        factors (list[torch.Tensor]): 
            - If tensor==True: [X] for symmetric or [X, Y, Z] for asymmetric.
            - If tensor==False: [X] for symmetric or [X, Y] for asymmetric.
        symmetric (bool): Whether the model is symmetric.
        tensor (bool): If True, CP–tensor formulas are used; otherwise matrix factorization.
    
    Returns:
        torch.Tensor: The result of applying the operator.
    """
    if tensor:
        if symmetric:
            X = factors[0]
            dX = update  # same shape as X
            XX = X.T @ X
            return 3 * (dX @ (XX * XX)) + 6 * (X @ ((dX.T @ X) * XX))
        else:
            X, Y, Z = factors
            shapes = [X.shape, Y.shape, Z.shape]
            dX, dY, dZ = split(update, shapes)
            XX = X.T @ X
            YY = Y.T @ Y
            ZZ = Z.T @ Z
            RES_X = dX @ (YY * ZZ) + X @ ((dY.T @ Y) * ZZ) + X @ (YY * (dZ.T @ Z))
            RES_Y = dY @ (XX * ZZ) + Y @ ((dX.T @ X) * ZZ) + Y @ (XX * (dZ.T @ Z))
            RES_Z = dZ @ (XX * YY) + Z @ ((dX.T @ X) * YY) + Z @ (XX * (dY.T @ Y))
            return torch.cat((RES_X.reshape(-1), RES_Y.reshape(-1), RES_Z.reshape(-1)))
    else:
        # Matrix factorization case.
        if symmetric:
            X = factors[0]
            g_mat = update.reshape(X.shape)
            return (2 * g_mat @ (X.T @ X) + 2 * X @ (g_mat.T @ X))
        else:
            X, Y = factors
            shapes = [X.shape, Y.shape]
            gx, gy = split(update, shapes)
            op_x = (X @ (gy.T @ Y) + gx @ (Y.T @ Y)).reshape(-1)
            op_y = (Y @ (gx.T @ X) + gy @ (X.T @ X)).reshape(-1)
            return torch.cat((op_x, op_y))


###############################################################################
# 5. Generic Conjugate–Gradient Solver
###############################################################################
def cg_solve(operator_fn, b, damping, max_iter=100, epsilon=1e-13):
    """
    Generic conjugate gradient solver.
    
    Solves A(x) + damping*x = b for x.
    
    Args:
        operator_fn (callable): A function mapping x to A(x) (same shape as x).
        b (torch.Tensor): Right-hand side.
        damping (float): Damping parameter.
        max_iter (int): Maximum iterations.
        epsilon (float): Tolerance on the residual norm.
    
    Returns:
        torch.Tensor: The solution vector x.
    """
    x = torch.zeros_like(b)
    r = b - (operator_fn(x) + damping * x)
    p = r.clone()
    rs_old = (r * r).sum()
    for i in range(max_iter):
        Ap = operator_fn(p) + damping * p
        alpha = rs_old / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if torch.sqrt(rs_new) <= epsilon:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


###############################################################################
# 6. Measurement Operator Class
###############################################################################
class GaussianLinearMeasurementOperator:
    def __init__(self, n1, n2, n3, m, identity=False, tensor=True):
        """
        For CP–tensor models (tensor=True) the underlying measurement tensor
        is of shape (m, n1, n2, n3). For matrix factorization (tensor=False), it is
        of shape (m, n1, n2) (with n2=n1 in the symmetric case).
        """
        self.identity = identity
        self.m = m
        self.tensor = tensor
        if tensor:
            self.n1 = n1; self.n2 = n2; self.n3 = n3
            shape = (m, n1, n2, n3)
        else:
            self.n1 = n1; self.n2 = n2  # for symmetric, n2 equals n1.
            shape = (m, n1, n2)
        self.A_tensors = torch.randn(*shape, device=device) / torch.sqrt(torch.tensor(m, device=device, dtype=torch.float64))

    def A(self, X):
        if self.identity:
            return X.flatten()
        else:
            # For CP–tensor: X has 3 indices; for matrix factorization: 2 indices.
            if self.tensor:
                return torch.einsum('ijkl,jkl->i', self.A_tensors, X)
            else:
                return torch.einsum('ijk,jk->i', self.A_tensors, X)

    def A_adj(self, y):
        if self.identity:
            # Identity operator: simply reshape.
            return y
        else:
            if self.tensor:
                y_expanded = y.reshape(self.m, 1, 1, 1)
                return torch.sum(y_expanded * self.A_tensors, dim=0)
            else:
                y_expanded = y.reshape(self.m, 1, 1)
                return torch.sum(y_expanded * self.A_tensors, dim=0)


###############################################################################
# 7. Unified Gradient Transpose Function
###############################################################################
def unified_nabla_c_transpose_g(factors, v, symmetric, tensor=True):
    """
    Computes (nabla c)^T * v for both CP–tensor and matrix factorization.
    
    For CP–tensor (tensor=True):
      - Symmetric: returns a tensor of shape (n, r) computed via three Einstein sums.
      - Asymmetric: returns a tuple (A, B, C).
    
    For matrix factorization (tensor=False):
      - Symmetric: returns ((v + v^T) @ X).reshape(-1)
      - Asymmetric: returns ( (v @ Y).reshape(-1), (v^T @ X).reshape(-1) )
    
    Args:
        factors (list[torch.Tensor]): See description.
        v (torch.Tensor): 
            For tensor==True: 
                • symmetric: shape (n, n, n)
                • asymmetric: shape (n1, n2, n3)
            For tensor==False:
                • symmetric: either a flattened or (n, n) tensor.
                • asymmetric: either flattened or of shape (n1, n2).
        symmetric (bool): Whether the model is symmetric.
        tensor (bool): If True, CP–tensor formulas are used; else matrix factorization.
    
    Returns:
        For symmetric: torch.Tensor of shape (n, r) or flattened.
        For asymmetric: tuple of tensors.
    """
    if tensor:
        X, Y, Z = factors
        A = torch.einsum('ijk,jl,kl->il', v, Y, Z)
        B = torch.einsum('ijk,il,kl->jl', v, X, Z)
        C = torch.einsum('ijk,il,jl->kl', v, X, Y)
        return A + B + C if symmetric else (A, B, C) 
    else:
        if symmetric:
            X = factors[0]
            if v.dim() == 1:
                v_mat = v.reshape(X.shape[0], X.shape[0])
            else:
                v_mat = v
            return ((v_mat + v_mat.T) @ X)
        else:
            X, Y = factors
            if v.dim() == 1:
                v_mat = v.reshape(X.shape[0], Y.shape[0])
            else:
                v_mat = v
            g_x = (v_mat @ Y)
            g_y = (v_mat.T @ X)
            return (g_x, g_y)


###############################################################################
# 8. Unified Run Methods Function
###############################################################################
def run_methods(methods_test, experiment_setups, n1, n2, n3, r_true, m, identity, device,
                n_iter, spectral_init, base_dir, loss_ord, initial_relative_error, symmetric,
                tensor=True, corr_level=0, q=0.9, lambda_=0.0001, gamma=0.001):
    """
    Runs the various descent methods.
    
    Depending on the Boolean flag "tensor", the reconstruction, initialization,
    gradient and operator routines are selected to handle either CP–tensor or
    matrix factorization.
    """
    # Initialize measurement operator.
    # For CP–tensor: use (n1,n2,n3); for matrix factorization:
    #   symmetric: use (n1,n1) and asymmetric: use (n1,n2)
    if tensor:
        if symmetric:
            measurement_operator = GaussianLinearMeasurementOperator(n1, n1, n1, m, identity=identity, tensor=True)
        else:
            measurement_operator = GaussianLinearMeasurementOperator(n1, n2, n3, m, identity=identity, tensor=True)
    else:
        if symmetric:
            measurement_operator = GaussianLinearMeasurementOperator(n1, n1, None, m, identity=identity, tensor=False)
        else:
            measurement_operator = GaussianLinearMeasurementOperator(n1, n2, None, m, identity=identity, tensor=False)
    
    outputs = dict()
    for experiment_setup in experiment_setups:
        r, kappa = experiment_setup
        print("=" * 80)
        print(f"Experiment Setup: r = {r}, kappa = {kappa}")
        print("=" * 80)
        # Construct ground‐truth factors.
        ux, _ = torch.linalg.qr(torch.rand(n1, r_true, device=device))
        vx, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
        singular_values = torch.linspace(1.0, 1/kappa, r_true, device=device)
        S = torch.diag(singular_values)
        X_star = ux @ S @ vx.T

        if tensor:
            if symmetric:
                T_star = cp_reconstruct(X_star, X_star, X_star)
            else:
                uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
                vy, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
                uz, _ = torch.linalg.qr(torch.rand(n3, r_true, device=device))
                vz, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
                Y_star = uy @ S @ vy.T
                Z_star = uz @ S @ vz.T
                T_star = cp_reconstruct(X_star, Y_star, Z_star)
        else:
            if symmetric:
                T_star = X_star @ X_star.T
            else:
                # For asymmetric matrix factorization, construct a second factor.
                uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
                Y_star = uy @ S @ vx.T
                T_star = X_star @ Y_star.T

        y_true = measurement_operator.A(T_star)
        # Inject noise if desired.
        if corr_level > 0:
            num_ones = int(y_true.shape[0] * corr_level)
            mask_indices = np.random.choice(y_true.shape[0], size=num_ones, replace=False)
            mask = torch.zeros(y_true.shape[0], device=device)
            mask[mask_indices] = 1
            noise = torch.randn(y_true.shape[0], device=device)
            y_true = y_true + torch.norm(y_true) * noise * mask
        else:
            y_true = y_true.to(device)

        # Unified local initialization.
        if tensor:
            if symmetric:
                new_factors = unified_local_init(T_star, [X_star], [n1], initial_relative_error, r, True, tensor=True)
                X0, Y0, Z0 = new_factors[0], new_factors[0], new_factors[0]
            else:
                new_factors = unified_local_init(T_star, [X_star, Y_star, Z_star], [n1, n2, n3],
                                                  initial_relative_error, r, False, tensor=True)
                X0, Y0, Z0 = new_factors
            sizes = [X0.shape, Y0.shape, Z0.shape]
        else:
            if symmetric:
                new_factors = unified_local_init(T_star, [X_star], [n1], initial_relative_error, r, True, tensor=False)
                X0 = new_factors[0]
            else:
                new_factors = unified_local_init(T_star, [X_star, Y_star], [n1, n2],
                                                  initial_relative_error, r, False, tensor=False)
                X0, Y0 = new_factors
            # For matrix factorization, record shapes.
            sizes = [X0.shape] if symmetric else [X0.shape, Y0.shape]

        for method in methods_test:
            print(f"\n{'-' * 80}\nStarting method: {method}\n{'-' * 80}")
            # Reset factors for each method.
            if tensor:
                if symmetric:
                    X = X0.clone()
                    Y = X.clone()
                    Z = X.clone()
                else:
                    X = X0.clone()
                    Y = Y0.clone()
                    Z = Z0.clone()
            else:
                if symmetric:
                    X = X0.clone()
                else:
                    X = X0.clone()
                    Y = Y0.clone()
            errs = []

            for k in range(n_iter):
                # Reconstruct the model.
                if tensor:
                    if symmetric:
                        T = cp_reconstruct(X, X, X)
                    else:
                        T = cp_reconstruct(X, Y, Z)
                else:
                    if symmetric:
                        T = X @ X.T
                    else:
                        T = X @ Y.T

                err = torch.norm(T - T_star)
                rel_err = err / torch.norm(T_star)
                if k % 20 == 0:
                    print(f"{method:^30} | Iteration: {k:03d} | Relative Error: {rel_err.item():.3e}")
                if rel_err < 1e-14:
                    errs += [1e-15 for _ in range(k, n_iter)]
                    break
                errs.append(rel_err.item())

                # Compute the residual.
                residual = measurement_operator.A(T) - y_true

                # Compute subgradient based on loss_ord.
                if loss_ord == 1:
                    subgradient_h = measurement_operator.A_adj(torch.sign(residual)).view(-1)
                    h_c_x = torch.sum(torch.abs(residual)).item()
                elif loss_ord == 0.5:
                    subgradient_h = measurement_operator.A_adj(residual / torch.norm(residual)).view(-1)
                    h_c_x = torch.norm(residual).item()
                elif loss_ord == 2:
                    subgradient_h = measurement_operator.A_adj(residual).view(-1)
                    h_c_x = 0.5 * (torch.norm(residual) ** 2).item()

                # Compute gradient using the unified gradient-transpose function.
                if tensor:
                    if symmetric:
                        grad = unified_nabla_c_transpose_g([X,X,X], subgradient_h.view(n1, n1, n1), True, tensor=True)
                    else:
                        gX, gY, gZ = unified_nabla_c_transpose_g([X, Y, Z], subgradient_h.view(n1, n2, n3), False, tensor=True)
                        grad = torch.cat((gX.reshape(-1), gY.reshape(-1), gZ.reshape(-1)))
                else:
                    if symmetric:
                        grad = unified_nabla_c_transpose_g([X,X], subgradient_h, True, tensor=False)
                    else:
                        gX, gY = unified_nabla_c_transpose_g([X, Y], subgradient_h, False, tensor=False)
                        grad = torch.cat((gX.reshape(-1), gY.reshape(-1)))

                # Choose update strategy.
                if method in ['Gradient descent', 'Subgradient descent']:
                    stepsize = h_c_x / (torch.norm(grad) ** 2)
                    preconditioned_grad = grad
                else:
                    damping_val = torch.sqrt(torch.tensor(h_c_x)) if loss_ord == 2 else h_c_x * 1e-5
                    damping_val = 0 if method == 'Gauss-Newton' else damping_val
                    # Define the operator for CG depending on model and symmetry.
                    if tensor:
                        if symmetric:
                            operator_fn = lambda x: unified_operator(x, [X, X, X], True, tensor=True)
                        else:
                            operator_fn = lambda x: unified_operator(x, [X, Y, Z], False, tensor=True)
                    else:
                        if symmetric:
                            operator_fn = lambda x: unified_operator(x, [X,X], True, tensor=False)
                        else:
                            operator_fn = lambda x: unified_operator(x, [X, Y], False, tensor=False)
                            
                    preconditioned_grad = cg_solve(operator_fn, grad, damping_val)
                    stepsize = h_c_x / (torch.dot(subgradient_h, subgradient_h))
                
                # Update the factors.
                if tensor:
                    if symmetric:
                        X = X - stepsize * preconditioned_grad
                    else:
                        prgx, prgy, prgz = split(preconditioned_grad, sizes)
                        X = X - stepsize * prgx
                        Y = Y - stepsize * prgy
                        Z = Z - stepsize * prgz
                else:
                    if symmetric:
                        X = X - stepsize * preconditioned_grad
                    else:
                        prgx, prgy = split(preconditioned_grad, sizes)
                        X = X - stepsize * prgx
                        Y = Y - stepsize * prgy

            print(f"Method '{method}' completed after {k+1} iterations with final relative error: {errs[-1]:.3e}\n")
            file_name = f'experiments/exp{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}_{method}_l_{loss_ord}_r*={r_true}_r={r}_condn={kappa}_trial_0.csv'
            full_path = os.path.join(base_dir, file_name)
            np.savetxt(full_path, np.array(errs), delimiter=',')
            outputs[method] = errs
    return outputs


###############################################################################
# Main Execution
###############################################################################
if __name__ == '__main__':
    # Parameter definitions.
    n1 = 100
    n2 = 100
    n3 = 20

    r_true = 2
    m = n1 * r_true * 20
    symmetric = True      # Set True for symmetric factorization.
    # Set tensor=True for CP–tensor and tensor=False for matrix factorization.
    tensor = False         
    
    identity = False
    spectral_init = False
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loss_ord = 1
    initial_relative_error = 10**-2
    n_iter = 500

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # experiment_setups: (overparameterization, condition number)
    experiment_setups = [(2,10), (4, 10)]
    methods = ['Subgradient descent', 'Levenberg–Marquardt (ours)']
    methods_test = methods

    # Run the methods.
    run_methods(methods_test, experiment_setups, n1, n2, n3, r_true, m, identity, device,
                n_iter, spectral_init, base_dir, loss_ord, initial_relative_error, symmetric,
                tensor=tensor)

    # Compute and plot errors.
    errs, stds = collect_compute_mean(experiment_setups, loss_ord, r_true, False, methods, 
                                      f'{"tensor" if tensor else "matrix"}{"sym" if symmetric else ""}'
                                       )
    plot_losses_with_styles(errs, stds, r_true, loss_ord, base_dir, 
                             (('Symmetric ' if symmetric else 'Asymmetric ') + 
                              ('Tensor' if tensor else 'Matrix')), 1)
