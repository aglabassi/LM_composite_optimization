import os
from functools import partial
from multiprocessing import Process
import numpy as np
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles,matrix_recovery_assymetric, trial_execution_matrix, collect_compute_mean
from utils import plot_transition_heatmap
import pickle  

def save(obj, filename):
    """Saves an object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")

def load(filename):
    """Loads an object from a file using pickle."""
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filename}")
    return obj

if __name__ == "__main__":
    run = True
    problem = 'Burer-Monteiro' #or 'Symmetric CP'
    #problem = 'Asymmetric Matrix'
    #problem = 'Symmetric CP'
    loss_ord = 1
    r_true = 2
    n_cpu = 1
    trials = 20
    n_trial_div_n_cpu = 1
    #os.system('rm experiments/expbm*.csv') if symmetric else os.system('rm experiments/expasymmetric*.csv') 
    T = 1000
    n = 40
    np.random.seed(42)
    r =5
    

    methods = ['Subgradient descent','Gauss-Newton', 'Levenberg–Marquardt (ours)']
    methods_test = methods
    methods_all = methods

    init_radius_ratio =10**-3
    end_ratio = 10**-8
    keys_all = [(2,1)] #keep one
    keys_test = keys_all
    
    d_trials = [ i*2*n for i in range(1,20)]
    cor_interval= 0.025
    corr_ranges = [ [l, l+cor_interval]  for l in np.arange(0, 0.5, cor_interval)]
    
    save_path = f'./exp1/{keys_test[0]}_{problem}.pkl'
    if run:
        success_matrixes = dict( (method, np.zeros((len(d_trials), len(corr_ranges)) )) for method in methods) 
        for i,d_trial in enumerate(d_trials):
            for j,corr_range in enumerate(corr_ranges):
                
                success_counters = dict((method, 0) for method in methods)
    
                for _ in range(trials):
                    corr_factor = (corr_range[1] -corr_range[0]) *np.random.rand() + corr_range[0]
                    d = d_trial
                   
                    if problem == 'Burer-Monteiro':
                        outputs = trial_execution_matrix(range(0, 1), 
                                                         n, r_true, 
                                                         d, keys_test, 
                                                         init_radius_ratio, 
                                                         T, loss_ord, 
                                                         './', 
                                                         methods_test, 
                                                         True,
                                                         False,
                                                         corr_factor, geom_decay=True)
                    elif problem == 'Asymmetric Matrix':
                        outputs = trial_execution_matrix(range(0, 1), 
                                                         n, r_true, 
                                                         d, keys_test, 
                                                         init_radius_ratio, 
                                                         T, loss_ord, 
                                                         './', 
                                                         methods_test, 
                                                         False,
                                                         False,
                                                         corr_factor, geom_decay=True)
                    elif problem == 'Symmetric CP':
                        outputs = run_methods(
                            methods_test, 
                            keys_test, n, r_true, d, False, 'cpu', 
                            T, False, './', 1, init_radius_ratio,
                            corr_level=corr_factor)
                        
                    
                    for method in methods:
                        if outputs[method][-1] <= end_ratio:
                            success_counters[method] +=1
                            
                for method in methods:
                    success_matrixes[method][i,j] = success_counters[method]/trials 
    
        # Save success matrix
       
        save(success_matrixes, save_path)

    # Load success matrix for verification or further processing
    loaded_success_matrixes = load(save_path)
    
    plot_transition_heatmap(
        success_matrixes=loaded_success_matrixes,
        d_trials=d_trials,
        n=n,
        base_dir="./exp1",  # or your desired directory
        keys = keys_test[0],
        problem=problem,
    )
