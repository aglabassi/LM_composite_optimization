import os
from functools import partial
from multiprocessing import Process
import numpy as np
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles,matrix_recovery_assymetric, trial_execution_matrix, collect_compute_mean



if __name__ == "__main__":
    
    #Matrix
    
    loss_ord = 2
    kappa = 1
    symmetric = True
    identity = False
    r_true = 2
    n_cpu = 1
    n_trial_div_n_cpu = 1
    os.system('rm experiments/expbm*.csv') if symmetric else os.system('rm experiments/expasymmetric*.csv') 
    T = 1000
    n = 20
    np.random.seed(42)
    
    if loss_ord == 2:
        methods = [ 'vanilla_gd', 'gn', 'pred_gd'] #smoooth BM
    else:
        methods =  [ 'vanilla_subgd', 'rebalance', 'gnp']

    init_radius_ratio = 0.1
    keys = [(2,1), (4,1), (2,100), (4,100)]
    
    d = 10 * n * r_true
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
