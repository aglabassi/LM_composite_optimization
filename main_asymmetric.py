import os
from functools import partial
from multiprocessing import Process
import numpy as np
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles,matrix_recovery_assymetric, trial_execution_matrix, collect_compute_mean



if __name__ == "__main__":
    
    #Matrix

    loss_ord = 1
    kappa = 1
    symmetric = False
    identity = False
    r_true = 2
    n_cpu = 1
    n_trial_div_n_cpu = 1
    #os.system('rm experiments/expbm*.csv') if symmetric else os.system('rm experiments/expasymmetric*.csv') 
    T = 500
    n = 50
    np.random.seed(42)
    r = 3
    if loss_ord == 2:
        if r == r_true:
            methods = [ 'Gradient descent', 'Scaled subgradient', 'Gauss-Newton', 'Levenberg–Marquard (ours)'] #smoooth assymetric
        else:
            methods = ['Gradient descent', 'Precond. gradient', 'Gauss-Newton'  ,'Levenberg–Marquard (ours)'] #smoooth assymetric over
    else:
        if r == r_true:
            methods = [ 'Subgradient descent' , 'Scaled subgradient', 'Gauss-Newton', 'Levenberg–Marquard (ours)']
        else:
            methods = [ 'Subgradient descent' , 'OPSA($\lambda=10^{-3}$)', 'OPSA($\lambda=10^{-8}$)', 'Gauss-Newton', 'Levenberg–Marquard (ours)']
            
    init_radius_ratio = 0.001
    keys = [(r,1), (r,10)]
    
    d = 20*n * r_true
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if n_cpu > 1:
        
        processes = [  Process(name=f"cpu {cpu}", target=partial(trial_execution_matrix, range(cpu*n_trial_div_n_cpu, (cpu+1)*n_trial_div_n_cpu), n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric, identity))
                    for cpu in range(n_cpu) ]  
    
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
    else:
        trial_execution_matrix(range(0, n_trial_div_n_cpu), n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric, identity)
    

    losses, stds = collect_compute_mean(keys, loss_ord, r_true, False, methods, 'bm' if symmetric else 'asymmetric' )
    plot_losses_with_styles(losses, stds, r_true, loss_ord, base_dir, "Burer-Monteiro" if symmetric else 'Asymmetric', kappa)
