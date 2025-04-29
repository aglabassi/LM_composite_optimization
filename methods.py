#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:19:02 2025

@author: aglabassi
"""

def subgradient_method(
    stepsize_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    x0,
    n_iter=1000
):
    """
    Subgradient method.

    Args:
        stepsize_fc: function (k, x) -> stepsize at iteration k.
        subgradient_fc: function (x) -> element of ∂v(F(x)).
        action_nabla_F_transpose_fc: function (x, v) -> ∇F(x)^T v.
        x0: starting point.
        n_iter: number of iterations.

    Returns:
        iterates
    """
    xs = []
    x = x0
    for k in range(n_iter):
        xs.append(x)
        stepsize = stepsize_fc(k, x)
        subgradient = subgradient_fc(x)
        gradient = action_nabla_F_transpose_fc(x, subgradient)

        x = x - stepsize * gradient

    return xs+[x]





def LM_subgradient_method(
    stepsize_fc,
    damping_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    levenberg_marquardt_linear_system_solver,
    x0,
    n_iter=1000
):
    """
    Levenberg-Marquardt subgradient method.

    Args:
        stepsize_fc: function (k, x) -> stepsize at iteration k.
        damping_fc: function (k, x) -> damping parameter at iteration k.
        subgradient_fc: function (x) -> element of ∂v(F(x)).
        action_nabla_F_transpose_fc: function (x, v) -> ∇F(x)^T v.
        levenberg_marquardt_linear_system_solver: 
            function (x, damping, b) -> solves (∇F(x)^T ∇F(x) + damping * I)g = b.
        x0: starting point.
        n_iter: number of iterations.

    Returns:
        iterates
    """
    xs = []
    x = x0
    for k in range(n_iter):
        xs.append(x)
        stepsize = stepsize_fc(k, x)
        damping = damping_fc(k, x)
        subgradient = subgradient_fc(x)
        gradient = action_nabla_F_transpose_fc(x, subgradient)
        preconditioned_gradient = levenberg_marquardt_linear_system_solver(x, damping, gradient)
        
        x = x - stepsize * preconditioned_gradient

    return xs+[x]





def GN_subgradient_method(
    stepsize_fc,
    subgradient_fc,
    action_nabla_F_transpose_fc,
    gauss_newton_linear_system_solver,
    x0,
    n_iter=1000
):
    """
    Gauss-newton subgradient method.

    Args:
        stepsize_fc: function (k, x) -> stepsize at iteration k.
        subgradient_fc: function (x) -> element of ∂v(F(x)).
        action_nabla_F_transpose_fc: function (x, v) -> ∇F(x)^T v.
        gauss_newton_linear_system_solver: 
            function (x, b) -> solves  (∇F(x)^T ∇F(x)) g = b.
        x0: starting point.
        n_iter: number of iterations.

    Returns:
        iterates
    """
    xs = []
    x = x0
    for k in range(n_iter):
        xs.append(x)
        stepsize = stepsize_fc(k, x)
        subgradient = subgradient_fc(x)
        gradient = action_nabla_F_transpose_fc(x, subgradient)
        preconditioned_gradient = gauss_newton_linear_system_solver(x, gradient)
        
        x = x - stepsize * preconditioned_gradient

    return xs+[x]



