#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:25:33 2025

@author: aglabassi
"""
import torch
import torch.nn.functional as F


###############################################################################
# Measurement Operator Class
###############################################################################
class LinearMeasurementOperator:
    def __init__(self, n1, n2, n3, m,  device, identity=False, tensor=True, distribution='Gaussian'):
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
        if not identity:
            if distribution == 'Gaussian':
                self.A_tensors = torch.randn(*shape, device=device) / torch.sqrt(torch.tensor(m, device=device, dtype=torch.float64))
            elif distribution == 'Bernoulli:':
                self.A_tensors = (torch.rand(*shape, device=device) < 0.5).to(torch.float64)

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



def local_init(T_star, factors, dims, tol, r, symmetric, tensor=True):
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
            T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[0], new_factors[0])
        else:
            T =torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[1], new_factors[2])
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
                T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[0], new_factors[0])
            else:
                T = torch.einsum('ir,jr,kr->ijk',new_factors[0], new_factors[1], new_factors[2])
        else:
            if symmetric:
                T = new_factors[0] @ new_factors[0].T
            else:
                T = new_factors[0] @ new_factors[1].T
        
        err_rel = torch.norm(T - T_star) / torch.norm(T_star)
    return new_factors


def generate_data_and_initialize(
    measurement_operator,
    n1, 
    r_true, 
    r,
    n2=None, 
    n3=None,
    device='cpu',
    kappa=10.0,
    corr_level=0.0,
    symmetric=False,
    tensor=False,
    initial_relative_error=1e-1
):
    """
    Generates synthetic data (matrix or tensor) along with noise,
    then performs a local initialization for factorization.

    Parameters
    ----------
    measurement_operator : object
        An operator that has a method A(...) that applies the measurement.
    n1 : int
        Dimension 1 for the ground-truth factor(s).
    r_true : int
        True rank of the ground-truth factor(s).
    r : int
        Desired rank for initialization.
    n2 : int, optional
        Dimension 2 (required for asymmetric matrix or 3D tensor if not symmetric).
    n3 : int, optional
        Dimension 3 (required for 3D tensor).
    device : str, optional
        Torch device (e.g., 'cpu' or 'cuda').
    kappa : float, optional
        Ratio controlling the range of singular values (1.0 to 1/kappa).
    corr_level : float, optional
        If > 0, fraction of measurements to corrupt with noise.
    symmetric : bool, optional
        Whether the factorization is symmetric.
    tensor : bool, optional
        Whether to construct a 3D tensor or a matrix.
    initial_relative_error : float, optional
        Relative error level for local_init(...).

    Returns
    -------
    T_star (for metric evaluation purposes)
    y_observed : torch.Tensor
        The measurement vector (possibly corrupted by noise).
    factors : tuple
        The initialized factor(s). For a 3D tensor, returns (X0, Y0, Z0).
        For a matrix, returns (X0, Y0)
    sizes : list of torch.Size
        Shapes of the returned factor(s).
    """

    # -- 1) Construct ground‐truth factors.
    # Orthonormal bases for X
    ux, _ = torch.linalg.qr(torch.rand(n1, r_true, device=device))
    vx, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
    singular_values = torch.linspace(1.0, 1.0 / kappa, r_true, device=device)
    S = torch.diag(singular_values)
    X_star = ux @ S @ vx.T  # shape: (n1, r_true)

    # We'll create either a tensor T_star (3D) or a matrix T_star (2D) depending on flags
    if tensor:
        # 3D tensor
        if symmetric:
            # Symmetric => same factor X_star used in each mode
            T_star = torch.einsum('ir,jr,kr->ijk', X_star, X_star, X_star)
        else:
            # Asymmetric => create distinct Y_star, Z_star
            if n2 is None or n3 is None:
                raise ValueError("n2 and n3 must be provided for a 3D asymmetric tensor.")
            uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
            vy, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))
            uz, _ = torch.linalg.qr(torch.rand(n3, r_true, device=device))
            vz, _ = torch.linalg.qr(torch.rand(r_true, r_true, device=device))

            Y_star = uy @ S @ vy.T
            Z_star = uz @ S @ vz.T
            T_star = torch.einsum('ir,jr,kr->ijk', X_star, Y_star, Z_star)
    else:
        # Matrix
        if symmetric:
            # Symmetric => T_star = X_star X_star^T
            T_star = X_star @ X_star.T
        else:
            # Asymmetric => create second factor Y_star
            if n2 is None:
                raise ValueError("n2 must be provided for asymmetric matrix factorization.")
            uy, _ = torch.linalg.qr(torch.rand(n2, r_true, device=device))
            Y_star = uy @ S @ vx.T
            T_star = X_star @ Y_star.T

    # Ensure T_star is on the correct device (usually already is, 
    # but this is a safety net):
    T_star = T_star.to(device)

    # -- 2) Compute measurements and possibly add noise.
    y_true = measurement_operator.A(T_star).to(device)

    # Generate y_false on the same device
    if not tensor:
        # For matrix case
        # Provide n2 if not symmetric
        shape_for_rand = (n1, n2) if n2 is not None else (n1, n1)
        y_false = measurement_operator.A(torch.rand(shape_for_rand, device=device))
    else:
        # For tensor case
        shape_for_rand = (n1, n2, n3)
        y_false = measurement_operator.A(torch.rand(shape_for_rand, device=device))

    num_ones = int(y_true.shape[0] * corr_level)
    perm = torch.randperm(y_true.shape[0], device=device)
    mask_indices = perm[:num_ones]

    mask = torch.zeros(y_true.shape[0], device=device)
    mask[mask_indices] = 1

    y_observed = (1 - mask) * y_true + mask * y_false

    # -- 3) Unified local initialization.
    # Note: we assume you have a function local_init(...) defined elsewhere.
    if tensor:
        if symmetric:
            new_factors = local_init(
                T_star, [X_star], [n1],
                initial_relative_error, r, True, tensor=True
            )
            X0 = Y0 = Z0 = new_factors[0]
            sizes = [X0.shape, Y0.shape, Z0.shape]
            factors = (X0, X0, X0)
        else:
            new_factors = local_init(
                T_star, [X_star, Y_star, Z_star], [n1, n2, n3],
                initial_relative_error, r, False, tensor=True
            )
            X0, Y0, Z0 = new_factors
            sizes = [X0.shape, Y0.shape, Z0.shape]
            factors = (X0, Y0, Z0)
    else:
        if symmetric:
            new_factors = local_init(
                T_star, [X_star], [n1],
                initial_relative_error, r, True, tensor=False
            )
            X0 = new_factors[0]
            sizes = [X0.shape]
            factors = (X0, X0)
        else:
            new_factors = local_init(
                T_star, [X_star, Y_star], [n1, n2],
                initial_relative_error, r, False, tensor=False
            )
            X0, Y0 = new_factors
            sizes = [X0.shape, Y0.shape]
            factors = (X0, Y0)

    return T_star, y_observed, factors, sizes




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
        numel = torch.prod(torch.tensor(shape))
        block = concatenated[start:start + numel].reshape(shape)
        output.append(block)
        start += numel
    return tuple(output)


###############################################################################
# Generic Conjugate–Gradient Solver
###############################################################################
def cg_solve(operator_fn, b, damping, max_iter=100, epsilon=1e-20):
    """
    Generic conjugate gradient solver.
    
    Solves operator(x) + damping*x = b for x, operator is a linear map.
    
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
        if rs_new <= epsilon**2:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def  nabla_F_transpose_g(factors, v, symmetric, tensor=True):
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

def compute_gradient(
    X, 
    Y, 
    Z, 
    subgradient_h, 
    n1, 
    n2, 
    n3, 
    symmetric=False, 
    tensor=False
):
    """
    Computes the gradient given:
      - Factor(s) X, Y, Z
      - A subgradient `subgradient_h`
      - A function nabla_F_transpose_g(...) that does the adjoint operation
      - The dimensions n1, n2, n3
      - Flags indicating if the problem is symmetric and/or tensor-based.

    Returns
    -------
    grad : torch.Tensor
        A 1D tensor containing the concatenated gradient.
    """

    if tensor:
        # 3D tensor
        if symmetric:
            # subgradient_h is shaped [n1, n1, n1] => ensure correct view
            grad = nabla_F_transpose_g([X, X, X],
                                       subgradient_h.view(n1, n1, n1),
                                       True, 
                                       tensor=True)
        else:
            # subgradient_h is shaped [n1, n2, n3]
            gX, gY, gZ = nabla_F_transpose_g([X, Y, Z],
                                             subgradient_h.view(n1, n2, n3),
                                             False, 
                                             tensor=True)
            grad = torch.cat((gX.reshape(-1), gY.reshape(-1), gZ.reshape(-1)))
    else:
        # Matrix
        if symmetric:
            # subgradient_h is shaped [n1, n1]
            grad = nabla_F_transpose_g([X, X],
                                       subgradient_h, 
                                       True, 
                                       tensor=False)
        else:
            # subgradient_h is shaped [n1, n2]
            gX, gY = nabla_F_transpose_g([X, Y],
                                         subgradient_h, 
                                         False, 
                                         tensor=False)
            grad = torch.cat((gX.reshape(-1), gY.reshape(-1)))
            

    return grad



def compute_stepsize_and_damping(
    method,
    grad,
    subgradient_h,
    h_c_x,
    loss_ord,
    symmetric,
    geom_decay=False,
    lambda_=None,
    q=None,
    gamma=None,
    k=None
):
    """
    Computes the stepsize and damping for various methods.

    Parameters
    ----------
    method : str
        The update method (e.g., 'Gradient descent', 'Scaled gradient', etc.).
    grad : torch.Tensor
        The gradient (flattened or otherwise). Used for 'Gradient descent'/'Subgradient descent'.
    subgradient_h : torch.Tensor
        The subgradient (same shape as needed in dot-product). Used in certain methods.
    h_c_x : float
        The scalar h(c(X, Y)) or a related objective value.
    loss_ord : int, optional
        The order of the loss function (only used for 'Levenberg-Marquardt (ours)' logic).
    geom_decay : bool, optional
        Flag to indicate geometric damping decay (for 'Levenberg-Marquardt (ours)' only).
    lambda_ : float, optional
        Base damping parameter if geom_decay is True.
    q : float, optional
        Decay rate if geom_decay is True.
    k : int, optional
        Current iteration index if geom_decay is True.

    Returns
    -------
    stepsize : float
        The computed stepsize to be used for the update.
    damping : float
        The damping parameter used in some preconditioned methods.
    """

    # Default damping if not used in the method
    damping = 0.0
    constant_stepsize = 1e0

    # -- 1) Plain (sub)gradient methods
    if method in ['Gradient descent', 'Subgradient descent']:
        stepsize = h_c_x / (torch.norm(grad) ** 2)

    # -- 2) Preconditioned or scaled methods
    else:
        if method == 'Scaled gradient($\lambda=10^{-8}$)':
            damping = 1e-5
            stepsize = constant_stepsize 

        elif method == 'Precond. gradient':
            # Example: damping depends on sqrt(h_c_x)
            damping = torch.sqrt(torch.tensor(h_c_x)) * 2.5e-3
            stepsize = constant_stepsize
            
        elif method  in ['Levenberg-Marquardt (ours)', 'Gauss-Newton']:
            # Damping depends on loss_ord, plus possibly geometric decay
            if geom_decay:
                # damping = lambda_ * (q ** k)
                # Ensure lambda_, q, k are not None
                if lambda_ is None or q is None or k is None:
                    raise ValueError("lambda_, q, k must be provided if geom_decay=True.")

                damping = lambda_ * (q ** k)
                stepsize= gamma * (q**k)
            else:
                # fallback if not geometric
                if loss_ord == 2:
                    damping = torch.sqrt(torch.tensor(h_c_x)) * 2.5e-3
                else:
                    damping = h_c_x * 1e-5

                stepsize = h_c_x / torch.dot(subgradient_h, subgradient_h)

        else:
            raise NotImplementedError(f"Unknown method: {method}")
    return stepsize, (damping if method != 'Gauss-Newton' else 0)


def update_factors(
    X, 
    Y, 
    Z, 
    preconditioned_grad, 
    stepsize, 
    sizes, 
    split_fn, 
    symmetric=False, 
    tensor=False
):
    """
    Updates the factors (X, Y, Z) in-place or via reassignment given a preconditioned gradient.
    
    Parameters
    ----------
    X : torch.Tensor
        Current factor for mode-1 (or the single factor if symmetric).
    Y : torch.Tensor or None
        Factor for mode-2 (or same as X if symmetric).
    Z : torch.Tensor or None
        Factor for mode-3 (used if tensor=True and not symmetric).
    preconditioned_grad : torch.Tensor
        The concatenated gradient, or the direct gradient (if symmetric).
    stepsize : float
        The scalar stepsize.
    sizes : list of torch.Size
        A list describing the shapes of each factor (e.g., [X.shape, Y.shape, Z.shape]).
    split_fn : callable
        A function that splits the flattened gradient into separate factor gradients.
        For instance: prgx, prgy, prgz = split_fn(preconditioned_grad, sizes).
    symmetric : bool
        If True, we assume a symmetric problem (X=Y=Z for a tensor).
    tensor : bool
        If True, indicates a 3D tensor problem; otherwise a 2D matrix problem.

    Returns
    -------
    X, Y, Z : torch.Tensor
        The updated factors. If symmetric, Y and Z will point to X.
    """

    if tensor:
        # 3D Tensor
        if symmetric:
            # Single factor used for all modes
            X = X - stepsize * preconditioned_grad
            Y = X
            Z = X
        else:
            # Split the preconditioned gradient into 3 parts
            prgx, prgy, prgz = split_fn(preconditioned_grad, sizes)
            X = X - stepsize * prgx
            Y = Y - stepsize * prgy
            Z = Z - stepsize * prgz
    else:
        # 2D Matrix
        if symmetric:
            # Single factor used for both rows/cols
            X = X - stepsize * preconditioned_grad
            Y = X
            Z = X  # Not strictly used in a matrix scenario, but kept for consistency
        else:
            # Split the preconditioned gradient into 2 parts
            prgx, prgy = split_fn(preconditioned_grad, sizes)
            X = X - stepsize * prgx
            Y = Y - stepsize * prgy

    return X, Y, Z



