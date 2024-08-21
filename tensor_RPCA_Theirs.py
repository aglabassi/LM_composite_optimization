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
from utils import thre

tl.set_backend('pytorch')


def rpca(G_true, factors_true, G0_init, factors0_init, X, Y, measurement_operator, ranks, z0, z1, eta, decay, T, epsilon, device, spectral_init, perturb=0.1,fix_G=False, skip=[]):
    
    torch.set_printoptions(precision=10)
    
    ## Initialization
    if spectral_init:
        G_t, factors_t = tucker(Y - thre(Y, z0, device), rank=ranks)
    else:
        G_t, factors_t = G0_init.clone(), [ f.clone() for f in factors0_init ]
        
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
        to_append = 1 if torch.isnan(torch.norm(X_t - X)) or torch.norm(X_t - X)/torch.norm(X) > 1000  else torch.norm(X_t - X).item()
        errs.append(to_append)
        
        
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
            factors_t_copy[k] = torch.eye(A_t.shape[1]).to(device).double()
            A_breve_t = tucker_to_unfolded((G_t, factors_t_copy), k).T 

            ATA_t_copy = ATA_t.copy()
            ATA_t_copy[k] = torch.eye(A_t.shape[1]).to(device).double()
            AbTAb_t = tucker_to_unfolded((G_t, ATA_t_copy), k) @ unfold(G_t, k).T

            #ker = torch.linalg.inv(AbTAb_t + epsilon * torch.eye(A_breve_t.shape[1]).to(device))
            A_t1 = (1 - eta) * A_t - eta * torch.linalg.solve(AbTAb_t + epsilon * torch.eye(A_breve_t.shape[1]).to(device).double(),  (unfold(D, k) @ A_breve_t).T).T
            factors_t1.append(A_t1)
        G_factors_t = []
        for k in range(order):
            if k in skip:
                G_factors_t.append(ATA_inverses_skipped[k] @ factors_t[k].T)
            else:
                G_factors_t.append(torch.linalg.inv(ATA_t[k] + epsilon * torch.eye(factors_t[k].shape[1]).to(device).double()) @ factors_t[k].T)
        G_t1 = G_t if fix_G else G_t - eta  * tucker_to_tensor((X_t + D, G_factors_t)) 
        factors_t = factors_t1
        G_t = G_t1
        
    return errs