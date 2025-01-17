import os
from functools import partial
from multiprocessing import Process
import numpy as np
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles,matrix_recovery_assymetric, trial_execution_matrix, collect_compute_mean
from utils import plot_transition_heatmap


if __name__ == "__main__":
    

    loss_ord = 1
    kappa = 1
    symmetric = True
    identity = False
    r_true = 2
    n_cpu = 1
    n_trial_div_n_cpu = 1
    #os.system('rm experiments/expbm*.csv') if symmetric else os.system('rm experiments/expasymmetric*.csv') 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    T = 350
    n = 5
    np.random.seed(42)
    r =2
    
    if loss_ord == 2:
        methods= ['Gradient descent', 'Scaled gradient($\lambda=10^{-8}$)', 'Precond. gradient', 'Levenberg–Marquardt (ours)']
        methods_test = methods
        methods_all =  methods     
    else:
        methods = [ 'Subgradient descent', 'Gauss-Newton', 'Levenberg–Marquardt (ours)']
        methods_test = methods
        methods_all = methods

    init_radius_ratio = 0.01
    keys_all = [(r,100)]
    keys_test = keys_all
    
    
    d_trials = [ i*2*n for i in range(6,6+8)]
    cor_interval= 0.02
    corr_ranges = [ [l, l+cor_interval]  for l in np.arange(0, 0.5, cor_interval)]
    
    success_matrixes = dict( (method, np.zeros((len(d_trials), len(corr_ranges)) )) for method in methods) 
    
    for i,d_trial in enumerate(d_trials):
        for j,corr_range in enumerate(corr_ranges):
            
            success_counters = dict((method, 0) for method in methods)
            trials = 10
            for _ in range(trials):
                corr_factor = (corr_range[1] -corr_range[0]) *np.random.rand() + corr_range[0]

                d = d_trial
                
                
                if n_cpu > 1:
                    
                    processes = [  Process(name=f"cpu {cpu}", target=partial(trial_execution_matrix, range(cpu*n_trial_div_n_cpu, (cpu+1)*n_trial_div_n_cpu), n, r_true, d, keys_test, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric, identity))
                                for cpu in range(n_cpu) ]  
                
                    
                    a = list(map(lambda p: p.start(), processes)) #run processes
                    b = list(map(lambda p: p.join(), processes)) #join processes
                else:
                    outputs = trial_execution_matrix(range(0, n_trial_div_n_cpu), n, r_true, d, keys_test, init_radius_ratio, T, loss_ord, base_dir, methods_test, symmetric, identity, corr_factor)
                
                
                for method in methods:
                    if outputs[method][-1] < 1e-13:
                        success_counters[method] +=1
                        
            for method in methods:
                success_matrixes[method][i,j] = success_counters[method]/trials 
            print(d_trial, corr_range, success_counters['Levenberg–Marquardt (ours)']/ trials)
            
    plot_transition_heatmap(
        success_matrixes=success_matrixes,
        d_trials=d_trials,
        n=n,
        base_dir=base_dir,  # or your desired directory
        problem='Burer-Monteiro'
)
    
