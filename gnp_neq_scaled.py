#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:15:59 2024

@author: aglabassi
"""

from sympy import symbols, Matrix, derive_by_array
import numpy as np

# Define symbols
x1, x2 = symbols('x1 x2')

# Define the vector x and matrix X
x = Matrix([x1, x2])

# Define matrix X (2x1)
X = Matrix([[x1], [x2]])

# Compute XX^T
XXT = X * X.T

# Vectorize XX^T
vec_XXT =np.array(XXT).reshape(-1)


J = derive_by_array(vec_XXT, x)


J_matrix = np.transpose(np.array(J).squeeze())

# Now you can safely compute J.T * J
C = Matrix(J_matrix.T @ J_matrix)

# Compute C * C * C
CCC = C * C * C

# This should give 0.
CCC_minus_C = (CCC-C).applyfunc(lambda x: x.simplify())


print("Product CCC minus C simplified:\n", CCC_minus_C.subs([(x1,1), (x2,1)]))
