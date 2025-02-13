#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:40:33 2025

@author: aglabassi
"""

import numpy as np

# -----------------------------
# Activation functions and derivatives
# -----------------------------
def sigmoid(x):
    """Elementwise logistic sigmoid."""
    return 1 / (1 + np.exp(-x))

def logsig_deriv(x):
    """
    Derivative wrt x of log(sigmoid(x)).

    If logsig(x) = log(sigmoid(x)), then
    d/dx logsig(x) = 1 - sigmoid(x).
    """
    return 1.0 - sigmoid(x)

def relu(x):
    """Elementwise ReLU."""
    return np.maximum(0, x)

def relu_deriv(x):
    """Derivative of ReLU; returns 1 if x>0, else 0."""
    return np.where(x > 0, 1.0, 0.0)

def inner_activation(x, inner_act="sigmoid"):
    """
    Computes the activation for the hidden layer.
    
    Args:
      x: input (numpy array)
      inner_act: a string, either "sigmoid" or "relu"
    """
    if inner_act == "sigmoid":
        return sigmoid(x)
    elif inner_act == "relu":
        return relu(x)
    else:
        raise ValueError(f"Unknown inner activation {inner_act}")

def inner_activation_deriv(x, inner_act="sigmoid"):
    """
    Computes the derivative of the inner activation, evaluated at x.
    
    For sigmoid, note that if X = sigmoid(x) then derivative = X*(1-X).
    For ReLU, derivative = 1 if x>0, 0 otherwise.
    """
    if inner_act == "sigmoid":
        s = sigmoid(x)
        return s * (1.0 - s)
    elif inner_act == "relu":
        return relu_deriv(x)
    else:
        raise ValueError(f"Unknown inner activation {inner_act}")

# -----------------------------------------
# Neural net cost function: outer activation is still sigmoid.
# -----------------------------------------
def c(A, W1, W2, inner_act="sigmoid"):
    """
    c(A,W1,W2) = log(sigmoid( inner_activation(A@W1) @ W2 ))
    
    Args:
      A: Input data.
      W1, W2: Network weights.
      inner_act: Activation for hidden layer ('sigmoid' or 'relu').
    """
    # 1) Y = A W1
    Y = A @ W1
    # 2) X = inner activation of Y
    X = inner_activation(Y, inner_act=inner_act)
    # 3) Z = X W2
    Z = X @ W2
    # 4) c = log(sigmoid(Z))
    return np.log(sigmoid(Z))

# -----------------------------------------
# Forward-Mode JVP: dot_c = J_c(W1,W2)*(dW1,dW2)
# -----------------------------------------
def nabla_c_action(A, W1, W2, dW1, dW2, inner_act="sigmoid"):
    """
    Forward-mode JVP for c(W1,W2) = log(sigmoid(inner_activation(AW1)W2)).
    Returns dot_c as an n-dimensional vector (shape (n,)).
    """
    # Forward pass:
    Y = A @ W1                 # shape (n,d2)
    X = inner_activation(Y, inner_act=inner_act)  # shape (n,d2)
    Z = X @ W2                 # shape (n,1)

    # Derivatives:
    dY = A @ dW1               # same shape as Y
    # For X = inner_activation(Y): dX = inner_activation_deriv(Y)*dY (elementwise)
    dX = inner_activation_deriv(Y, inner_act=inner_act) * dY
    # For Z = X W2: dZ = (dX @ W2) + (X @ dW2)
    dZ = (dX @ W2) + (X @ dW2)

    # For c = log(sigmoid(Z)), derivative wrt Z is (1 - sigmoid(Z)):
    logsigp_Z = logsig_deriv(Z)  # shape (n,1)
    dot_c = logsigp_Z * dZ       # elementwise multiply

    return dot_c.ravel()

# -----------------------------------------
# Reverse-Mode VJP: (dW1, dW2) = J_c(W1,W2)^T * v
# -----------------------------------------
def nabla_c_transpose_action(A, W1, W2, v, inner_act="sigmoid"):
    """
    Reverse-mode VJP for c(W1,W2) = log(sigmoid(inner_activation(AW1)W2)).
    That is, (dW1, dW2) = J_c(W1,W2)^T * v, for v in R^n.
    """
    # Forward pass to get intermediates:
    Y = A @ W1          # shape (n,d2)
    X = inner_activation(Y, inner_act=inner_act)      # shape (n,d2)
    Z = X @ W2          # shape (n,1)

    # For log(sigmoid(Z)): derivative wrt Z is (1 - sigmoid(Z))
    sZ = sigmoid(Z)                              # shape (n,1)
    deltaZ = v.reshape(-1,1) * (1.0 - sZ)         # shape (n,1)

    # Backprop through Z -> X:
    dW2 = X.T @ deltaZ                            # shape (d2,1)
    deltaX = deltaZ @ W2.T                        # shape (n,d2)

    # Backprop through X -> Y: using derivative of inner activation
    deltaY = deltaX * inner_activation_deriv(Y, inner_act=inner_act)  # shape (n,d2)

    dW1 = A.T @ deltaY                            # shape (d1,d2)

    return dW1, dW2

# A convenience to combine forward- then reverse-mode:
def nabla_c_transpose_dot_nabla_c_action(A, W1, W2, dW1, dW2, inner_act="sigmoid"):
    """
    Applies J_c(W1,W2)^T to the result of J_c(W1,W2)*(dW1, dW2).
    """
    return nabla_c_transpose_action(
        A, W1, W2,
        nabla_c_action(A, W1, W2, dW1, dW2, inner_act=inner_act),
        inner_act=inner_act
    )

# --------------------------------------------------
# Conjugate Gradient solver that uses the above
# Hessian-vector product approximation
# --------------------------------------------------
def compute_preconditionner_applied_to_2d_nn(A, W1, W2, gW1, gW2, 
                                             damping, 
                                             max_iter=100, 
                                             epsilon=1e-13,
                                             inner_act="sigmoid"):
    """
    Conjugate Gradient method to approximately solve:
        (J_c^T J_c + damping * I) x = g
    for a 2D neural net (single hidden layer).
    """

    def operator(X, Y, gx, gy):
        # Reshape the parameter perturbations:
        gx_mat = gx.reshape(X.shape)
        gy_mat = gy.reshape(Y.shape)
        # Apply (J_c^T J_c)(gx_mat, gy_mat):
        dW1_, dW2_ = nabla_c_transpose_dot_nabla_c_action(A, X, Y, gx_mat, gy_mat, inner_act=inner_act)
        return np.hstack((dW1_.ravel(), dW2_.ravel()))

    # Copy the current parameters:
    X = W1.copy()
    Y = W2.copy()
    # Flatten the gradient:
    gx = gW1.ravel()
    gy = gW2.ravel()

    # Initial guess is zero
    x = np.zeros_like(gx)
    y = np.zeros_like(gy)
    inp = np.hstack((x, y))

    # Right-hand side is the negative gradient (or 'g') you want to solve for
    g = np.hstack((gx, gy))

    # Residual: r = g - (J_c^T J_c + damping I) inp
    r = g - (operator(X, Y, x, y) + damping * inp)
    p = r.copy()
    rs_old = r @ r

    info = {'iterations': 0, 'residual_norm': np.sqrt(rs_old)}

    for i in range(max_iter):
        Ap = operator(X, Y, p[:len(x)], p[len(x):]) + damping * p
        alpha = rs_old / (p @ Ap)
        inp += alpha * p
        x = inp[:len(gx)]
        y = inp[len(gx):]
        r -= alpha * Ap

        rs_new = r @ r
        info['iterations'] = i + 1
        info['residual_norm'] = np.sqrt(rs_new)
        if info['residual_norm'] <= epsilon:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    # Reshape solution back to W1, W2 shapes
    return x.reshape(W1.shape), y.reshape(W2.shape)


# Binary cross entropy
def iterative_method(W10, W20, A, b, K, method, q=0.9, lambda_=1e-2, inner_act="sigmoid"):
    """
    W10, W20: initial weights (numpy arrays)
    A, b    : training data (A) and labels (b)
    K       : number of iterations
    method  : which method to use ('Gauss-Newton', 'LM', 'adam', or plain GD)
    q       : factor for damping decay in Levenberg-Marquardt
    lambda_ : initial damping for LM
    inner_act: inner activation ('sigmoid' or 'relu')
    """

    # Define binary cross-entropy "cost" pieces:
    def h(z):
        # Here: z = c(A,W1,W2) = log(sigmoid(XW2)) ...
        # Minimizing negative log-likelihood:
        #   - ∑ [y_i * log p_i + (1-y_i)*log(1-p_i)]
        # Here it looks like you have a custom expression with -np.inner(2*b - 1, z)
        return -1 * np.inner(2*b - 1, z)

    def grad_h(z):
        # derivative wrt z => -1*(2*b -1)
        return -1*(2*b - 1)
    
    # Copy initial parameters
    W1 = W10.copy()
    W2 = W20.copy()

    # If you want to handle Adam, set Adam hyperparameters
    alpha  = 1e-3   # learning rate
    beta1  = 0.9
    beta2  = 0.999
    eps    = 1e-8   # small constant for numerical stability

    # Adam moment buffers (first & second moments)
    m_w1 = np.zeros_like(W1)
    m_w2 = np.zeros_like(W2)
    v_w1 = np.zeros_like(W1)
    v_w2 = np.zeros_like(W2)

    accs = []
    for k in range(K):
        # Forward pass to get predictions:
        z = c(A, W1, W2, inner_act=inner_act).squeeze()  
        
        # Print cost (optional)
        if k%100 == 0:
            print(h(z))
        
        # Compute accuracy (here using the exp(c(.)) to map back to probabilities)
        pred = np.rint(np.exp(c(A, W1, W2, inner_act=inner_act))).reshape(-1)
        accuracy_k = np.sum(b == pred) / len(b)

        accs.append(h(z))
        
        # Outer gradient wrt c(A,W1,W2)
        outer_grad = grad_h(z)  # shape (n,)
        
        # Backprop: gradient wrt W1, W2
        grad_w1, grad_w2 = nabla_c_transpose_action(A, W1, W2, outer_grad, inner_act=inner_act)
        
        # Optionally apply Gauss-Newton / Levenberg-Marquardt preconditioner
        if method in ['Gauss-Newton', 'LM']:
            damping = 0.0 if (method == 'Gauss-Newton') else lambda_ * (q**k)
            grad_w1, grad_w2 = compute_preconditionner_applied_to_2d_nn(
                A, W1, W2, grad_w1, grad_w2, damping, inner_act=inner_act
            )
        
        # ------------------------------
        # ADAM update
        # ------------------------------
        if method == 'adam':
            # Update biased first moment estimate:
            m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
            m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2
            
            # Update biased second moment estimate:
            v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1 ** 2)
            v_w2 = beta2 * v_w2 + (1 - beta2) * (grad_w2 ** 2)
            
            # Correct bias in first & second moment
            m_w1_hat = m_w1 / (1 - beta1**(k+1))
            m_w2_hat = m_w2 / (1 - beta1**(k+1))
            v_w1_hat = v_w1 / (1 - beta2**(k+1))
            v_w2_hat = v_w2 / (1 - beta2**(k+1))
            
            # Update parameters
            W1 -= alpha * m_w1_hat / (np.sqrt(v_w1_hat) + eps)
            W2 -= alpha * m_w2_hat / (np.sqrt(v_w2_hat) + eps)
        
        # ------------------------------
        # Plain Gradient Descent update
        # (if not GN/LM or Adam)
        # ------------------------------
        else:
            stepsize = 0.001
            W1 = W1 - stepsize * grad_w1
            W2 = W2 - stepsize * grad_w2

    return accs

def generate_ill_conditioned_data(n=8, d1=4, d2=3, kappa=100.0):
    """
    Generates:
      - An ill-conditioned matrix A of shape (n, d1),
      - A vector b of length n (half +1, half 0),
      - (Large-magnitude parameters W1_star (d1 x d2) and W2_star (d2 x 1)
         can be defined as needed.)
    which can form a challenging, ill-conditioned neural-net problem.
    
    Args:
      n    : number of data points (rows of A)
      d1   : input dimension for W1
      d2   : hidden dimension for W1 and dimension of W2
      kappa: condition number factor
      
    Returns:
      A        : (n, d1) ill-conditioned
      b        : (n,) with half +1, half 0 (shuffled)
    """
    np.random.seed(0)

    # 1. Create a random (n x d1) matrix
    A_rand = np.random.randn(n, d1)

    # 2. Force ill-conditioning by modifying singular values
    #    We'll take SVD of A_rand and replace the singular values
    U, s, Vt = np.linalg.svd(A_rand, full_matrices=True)

    # Example: let singular values decay from 1 down to 1/kappa (log-spaced)
    s_new = np.logspace(np.log10(1.0), np.log10(1.0/kappa), min(n,d1))
    A_ill = (U[:,:min(n,d1)] * s_new) @ (Vt[:min(n,d1),])  # shape (n, d1)

    # 3. Construct b with half +1, half 0
    b = np.array([+1]*(n//2) + [0]*(n - n//2))
    np.random.shuffle(b)  # randomize the order

    return A_ill, b

# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Dimensions
    n, d1, d2 = 100, 10, 100
    
    q = 0.98
    lambda_ = 10**-10

    # Example data
    A, b  = generate_ill_conditioned_data(n, d1, d2, kappa=100)
    
    W1 = np.random.randn(d1, d2)
    W2 = np.random.randn(d2, 1)

    # Choose one of the following: 'SG', 'LM', or 'adam'
    methods = ['SG', 'LM', 'adam']
    ress = dict((method, []) for method in methods)
    
    # To switch inner activation, change inner_act below (e.g., "relu")
    inner_act = "sigmoid"

    for method in methods:
        ress[method] = iterative_method(W1, W2, A, b, 5000, method, q=q, lambda_=lambda_, inner_act=inner_act)
        print('-------')
        
    import matplotlib.pyplot as plt
    
    for method in methods:
        accuracies = ress[method]  # e.g., a list or numpy array of losses
        plt.plot(accuracies, label=method)
    
    plt.xlabel("Iteration")
    plt.ylabel("Loss (binary cross entropy)")
    plt.title("Loss vs. iteration for different methods")
    plt.legend()
    plt.show()
