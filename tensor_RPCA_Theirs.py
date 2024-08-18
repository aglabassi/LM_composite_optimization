#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:18:47 2024

@author: Harry Dong
"""

import torch
import tensorly as tl
from tensorly import tucker_to_tensor, tucker_to_unfolded, unfold
from tensorly.decomposition import tucker
from utils import random_perturbation, thre

tl.set_backend('pytorch')


def rpca(G_true, factors_true, X, Y, ranks, z0, z1, eta, decay, T, epsilon, device, spectral_init, perturb=0.1, skip=[]):
    
    torch.set_printoptions(precision=10)


    ## Initialization
    if spectral_init:
        G_t, factors_t = tucker(Y - thre(Y, z0, device), rank=ranks)
    else:
        G_t, factors_t = G_true + random_perturbation(G_true.shape, perturb), [ f + random_perturbation(f.shape, perturb)  for f in factors_true ]
        
    order = len(ranks)

    ATA_inverses_skipped = dict()
    ATA_skipped = dict()
    for k in skip:
        ATA_skipped[k] = factors_t[k].T @ factors_t[k]
        ATA_inverses_skipped[k] = torch.linalg.inv(ATA_skipped[k]) 
        
    errs = []
    ## Main Loop in ScaledGD RPCA
    for t in range(T):
        X_t = tucker_to_tensor((G_t, factors_t))
        S_t1 = thre(Y- X_t, z1 * (decay**t), device)
        print(torch.norm(X_t - X))
        
        
        errs.append(torch.norm( X_t - X ).item())
        
        
        factors_t1 = []
        D = S_t1 - Y
        ATA_t = []
        for k in range(order):
            if k in skip:
                ATA_t.append(ATA_skipped[k])
            else:
                ATA_t.append(factors_t[k].T @ factors_t[k])

        for k in range(order):
            if k in skip:
                factors_t1.append(factors_t[k])
                continue 
            
            A_t = factors_t[k]
            factors_t_copy = factors_t.copy()
            factors_t_copy[k] = torch.eye(A_t.shape[1]).to(device)
            A_breve_t = tucker_to_unfolded((G_t, factors_t_copy), k).T 

            ATA_t_copy = ATA_t.copy()
            ATA_t_copy[k] = torch.eye(A_t.shape[1]).to(device)
            AbTAb_t = tucker_to_unfolded((G_t, ATA_t_copy), k) @ unfold(G_t, k).T

            #ker = torch.linalg.inv(AbTAb_t + epsilon * torch.eye(A_breve_t.shape[1]).to(device))
            A_t1 = (1 - eta) * A_t - eta * torch.linalg.solve(AbTAb_t + epsilon * torch.eye(A_breve_t.shape[1]).to(device),  (unfold(D, k) @ A_breve_t).T).T
            factors_t1.append(A_t1)
        G_factors_t = []
        for k in range(order):
            if k in skip:
                G_factors_t.append(ATA_inverses_skipped[k] @ factors_t[k].T)
            else:
                G_factors_t.append(torch.linalg.inv(ATA_t[k] + epsilon * torch.eye(factors_t[k].shape[1]).to(device)) @ factors_t[k].T)
        G_t1 = G_t - eta  * tucker_to_tensor((X_t + D, G_factors_t))
        factors_t = factors_t1
        G_t = G_t1
    
    return errs