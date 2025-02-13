import os
import sys
from functools import partial
from multiprocessing import Process
import numpy as np
from matrix_utils import trial_execution_matrix


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from LM_composite_optimization.utils import plot_losses_with_styles, collect_compute_mean

if __name__ == "__main__":
    
    #Matrix
    
    loss_ord = 1 #0.5 => L2 norm not squared. 2 => L2 norm squared. 1 => L1 norm
    kappa = 1
    symmetric = True
    identity = False
    r_true = 2
    n_cpu = 1
    n_trial_div_n_cpu = 1
    #os.system('rm experiments/expbm*.csv') if symmetric else os.system('rm experiments/expasymmetric*.csv') 
    T = 500
    n = 10
    np.random.seed(42)
    r =5
    if loss_ord == 2:
        methods= [ 'Gradient descent','Scaled gradient($\lambda=10^{-8}$)', 'Precond. gradient', 'Levenberg–Marquardt (ours)'] if symmetric \
            else  [ 'Gradient descent','OPSA($\lambda=10^{-8}$)', 'Levenberg–Marquardt (ours)']
        methods_test = methods
        methods_all =  methods     
    else:
        methods = [  'Subgradient descent', 'Levenberg–Marquardt (ours)']
        methods_test = methods
        methods_all = methods


    init_radius_ratio = 0.01
    keys_all = [(r_true, 1), (r_true, 100), (r, 1), (r,100) ]
    
    keys_test = keys_all
    
    d = 20*n * r_true
    geom_decay = False
    base_dir = os.path.join(repo_root, f'LM_composite_optimization/experiment_results/{"polyak" if not geom_decay else "geometric_decaying"}')
    
    if n_cpu > 1:
        
        processes = [  Process(name=f"cpu {cpu}", target=partial(trial_execution_matrix, range(cpu*n_trial_div_n_cpu, (cpu+1)*n_trial_div_n_cpu), n, r_true, d,
                                                                 keys_test, init_radius_ratio, 
                                                                 T, loss_ord, base_dir, methods, symmetric,
                                                                 identity, geom_decay=geom_decay))
                    for cpu in range(n_cpu) ]  
    
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
    else:
        trial_execution_matrix(range(0, n_trial_div_n_cpu), n, r_true, d, keys_test, init_radius_ratio, T, 
                               loss_ord, base_dir, methods_test, symmetric, identity, geom_decay)
    

    losses, stds = collect_compute_mean(keys_all, loss_ord, r_true, False, methods_all, 'bm' if symmetric else 'asymmetric', base_dir)
    plot_losses_with_styles(losses, stds, r_true, loss_ord, base_dir, "Burer-Monteiro" if symmetric else 'Asymmetric', kappa)