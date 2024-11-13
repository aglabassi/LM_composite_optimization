#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:37:52 2024

@author: aglabassi
"""

import numpy as np
n=10
X = np.random.rand(10,5)
U,sigma, V_T = np.linalg.svd(X)