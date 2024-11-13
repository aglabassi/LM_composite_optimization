import numpy as np
import os
from utils import collect_compute_mean, plot_losses_with_styles

def generate_point_in_b_epsilon(p, epsilon):
    # Generate a random direction vector with the same dimension as p
    random_direction = np.random.randn(*p.shape)
    random_direction /= np.linalg.norm(random_direction)  # Normalize to unit vector
    
    # Generate a random distance within the epsilon radius
    distance = np.random.uniform(0, epsilon)
    
    # Create the new point within B_epsilon of p
    new_point = p + distance * random_direction
    
    return new_point

# Example usage
def generate_rip_matrix(m, n):
    """
    Generates an m x n random matrix with entries drawn from a standard normal distribution.
    Such matrices are likely to satisfy the RIP condition for compressed sensing applications.

    Parameters:
    m (int): Number of rows (measurements).
    n (int): Number of columns (dimension of the signal).

    Returns:
    np.ndarray: An m x n random Gaussian matrix.
    """
    # Generate a random Gaussian matrix with entries from N(0, 1/sqrt(m))
    A = np.random.randn(m, n) / np.sqrt(m)
    return A


r_true = 40

kappa = 1
keys = [(40,kappa), (100,kappa)]
n_trial = 100

#os.system('rm experiments/exphad*.csv')
for trial in range(n_trial):
    for r , kappa in keys:
        
        m = 10*r
        n = r
            
        
        x_star = np.concatenate((np.linspace(1, 1/kappa, r_true), np.zeros(n-r_true)))
        print(x_star)
        radius = 5
        x0 = generate_point_in_b_epsilon(x_star, radius)
        T = 200
        A = generate_rip_matrix(m, n)
        b = A@(x_star**2)
        methods = ['vanilla_subgradient', 'projected_subgradient', 'gnp']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        for method in methods:
            x = x0.copy() 
            z = x**2
            file_name = f'experiments/exphad_{method}_l_{1}_r*={r_true}_r={n}_condn={kappa}_trial_{trial}.csv'
            errs = []
            for i in range(T):
                damping = np.linalg.norm(x**2 - x_star**2)
                errs.append( damping )
                jac = 2*np.diag(x)
                v =  A.T@np.sign(  A@(x**2) - b )
                gradient = jac.T @ v
                print(method)
                print(damping)
                if method == 'vanilla_subgradient':
                    stepsize = np.linalg.norm( A@(x**2) - b, ord=1) / np.sum(gradient**2)
                    x = x - stepsize*gradient
                elif method == 'gnp':
                    stepsize =  np.linalg.norm( A@(x**2) - b, ord=1) / np.sum(v**2)
                    x = x - stepsize * np.linalg.solve( ( jac.T@jac +  damping*np.eye(n)), jac.T@v)
                
                elif method == 'projected_subgradient':
                    # Compute the stepsize
                    stepsize = np.linalg.norm(A @ z  - b, ord=1) / np.sum(v**2)
                    # Perform the subgradient step
                    z = z - stepsize * v
                    # Project onto the positive orthant
                    z = np.maximum(z, 0)
                    x = np.sqrt(z)

   
                full_path = os.path.join(base_dir, file_name)
                np.savetxt(full_path, np.array(errs)/errs[0], delimiter=',') 
    
           
losses, stds = collect_compute_mean(keys, 1, r_true, False, methods, 'had' )
plot_losses_with_styles(losses, stds, r_true, 1, base_dir, 'Hadamard', kappa)
       
        