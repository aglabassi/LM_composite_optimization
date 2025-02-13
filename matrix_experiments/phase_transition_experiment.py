import os
import sys
import numpy as np


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from LM_composite_optimization.utils import plot_transition_heatmap
from matrix_utils import trial_execution_matrix

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
    T = 1
    n = 40
    np.random.seed(42)
    r =5
    

    methods = ['Subgradient descent','Gauss-Newton', 'Levenberg–Marquardt (ours)']
    methods_test = methods
    methods_all = methods

    init_radius_ratio =10**-3
    end_ratio = 10**-8
    keys_all = [(5,100)] #keep one 
    keys_test = keys_all
    
    d_trials = [ i*2*n for i in range(1,20)]
    cor_interval= 0.025
    corr_ranges = [ [l, l+cor_interval]  for l in np.arange(0, 0.5, cor_interval)]
    
    
    base_dir = os.path.join(repo_root, 'LM_composite_optimization/experiment_results/phase_transition')
    save_path = os.path.join(base_dir,f'{keys_test[0]}_{problem}.pkl' )
    
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
                                                         base_dir, 
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
                                                         base_dir, 
                                                         methods_test, 
                                                         False,
                                                         False, geom_decay=True)
                        
                    
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
        base_dir=base_dir,  # or your desired directory
        keys = keys_test[0],
        problem=problem,
    )
