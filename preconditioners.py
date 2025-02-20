# -*- coding: utf-8 -*-
#Author: Abdel Ghani Labassi
import torch
from utils import split 

def LM_preconditioner(update, factors, symmetric, tensor=True):
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

def scaled_preconditioner(update, factors, symmetric, tensor=True):
    
    if tensor:
        raise NotImplementedError('Scaled does not exist for CP factorization')
    else:
        # Matrix factorization case.
        if symmetric:
            X = factors[0]
            g_mat = update.reshape(X.shape)
            return 2 * g_mat @ (X.T @ X)
        else:
            X, Y = factors
            shapes = [X.shape, Y.shape]
            gx, gy = split(update, shapes)
            op_x = (gx @ (Y.T @ Y)).reshape(-1)
            op_y = (gy @ (X.T @ X)).reshape(-1)
            return torch.cat((op_x, op_y))
