import numpy as np
import os
import sys

from result_processing_methods import plot_losses_with_styles, collect_compute_mean

def generate_point_on_boundary_positive_orthant(p, epsilon):
    """
    Generate a random point on the boundary of the ball
         B(p, ||p|| * epsilon)
    that lies in the positive orthant.
    
    Parameters:
      p       : numpy array of shape (n,), the center of the ball
                (assumed to be in the positive orthant)
      epsilon : float, the relative radius (scaled by ||p||)
      
    Returns:
      new_point : numpy array of shape (n,), a point on the boundary of
                  B(p, ||p|| * epsilon) that is in the positive orthant.
    """
    # Compute the radius of the ball
    radius = np.linalg.norm(p) * epsilon
    
    # Generate a random direction vector with positive entries.
    # Start with an n-dimensional standard normal vector,
    # take absolute value so that all coordinates are nonnegative,
    # and then normalize it.
    random_direction = np.abs(np.random.randn(*p.shape))
    random_direction /= np.linalg.norm(random_direction)
    
    # Place the new point exactly on the boundary
    new_point = p + radius * random_direction
    
    return new_point

# Example usage
def generate_difficult_A(m, n, kappa_A=2):
    """
    Generate an m x n matrix A with prescribed singular values for a hard nonnegative least squares problem.
    
    """
    # Create the singular values vector:
    s_values = np.linspace(1, 1/kappa_A, n)
    
    # Generate random orthonormal matrices U (m x m) and V (n x n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Build the Sigma matrix (m x n) with the singular values along the diagonal
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = s_values[i]
    
    # Construct A = U * Sigma * V^T
    A = U @ Sigma @ V.T
    
    return A

r_true = 10
r=100

kappa = 1
keys = [(r_true,1),(r_true,100), (r,1),(r,100)]
n_trial = 1

for trial in range(n_trial):
    for r , kappa in keys:
        
        m = 50*r
        n = r
            
        
        x_star = np.concatenate((np.linspace(1, 1/kappa, r_true), np.zeros(n-r_true)))
        radius = 1
        z0 = generate_point_on_boundary_positive_orthant(x_star**2, radius)
        x0 = np.sqrt(z0)
        T = 100
        A = generate_difficult_A(m, n)
        b = A@(x_star**2)
        methods = ['Polyak Subgradient', 'Gauss-Newton', 'Levenberg-Marquardt (ours)']
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results/polyak')
        os.makedirs(base_dir, exist_ok=True)
        
        
        for method in methods:
            x = x0.copy() 
            z = x**2
            file_name = f'exphad_{method}_l_{1}_r*={r_true}_r={n}_condn={kappa}_trial_{trial}.csv'
            errs = []
            for i in range(T):
                damping = np.linalg.norm( A@(x**2) - b, ord=1)/100
                
                if  np.isnan(damping) or damping > 10**2:
                    errs.append(1)
                    continue
                
                errs.append( damping/ np.linalg.norm(x_star**2) )
                    
                jac = 2*np.diag(x)
                v =  A.T@np.sign(  A@(x**2) - b )
                gradient = jac.T @ v
                print(method)
                print(damping)
                if method == 'Polyak Subgradient':
                    stepsize = np.linalg.norm( A@(x**2) - b, ord=1) / np.sum(gradient**2)
                    x = x - stepsize*gradient
                elif method in ['Gauss-Newton' , 'Levenberg-Marquardt (ours)']:
                    stepsize =  np.linalg.norm( A@(x**2) - b, ord=1) / np.sum(v**2)
                    damping = 0 if method == 'Gauss-Newton' else damping
                    try:
                        preconditioned_g, _ , _, _ = np.linalg.lstsq( jac.T@jac +  damping*np.eye(n) , jac.T@v, rcond=-1)
                    except:
                        preconditioned_g = jac.T@v
                    x = x - stepsize * preconditioned_g
                
                elif method == 'projected_subgradient':
                    # Compute the stepsize
                    stepsize = np.linalg.norm(A @ z  - b, ord=1) / np.sum(v**2)
                    # Perform the subgradient step
                    z = z - stepsize * v
                    # Project onto the positive orthant
                    z = np.maximum(z, 0)
                    x = np.sqrt(z)

   
                full_path = os.path.join(base_dir, file_name)
                np.savetxt(full_path, np.array(errs), delimiter=',') 
    
           
losses, stds = collect_compute_mean(keys, 1, r_true, False, methods, 'had', base_dir )
plot_losses_with_styles(losses, stds, r_true, 1, base_dir, 'Hadamard', kappa, had=True)
       
        