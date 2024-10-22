import sys,os
from functools import partial
from multiprocessing import Process
import numpy as np
import glob
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles,matrix_recovery_assymetric

def trial_execution(trials, n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric=True):
    
    for trial in trials:
        A, A_adj = create_rip_transform(n, d)
    
        for r, cond_number in keys:
            X_true = generate_matrix_with_condition(n, r_true, cond_number)
            Y_true = generate_matrix_with_condition(n, r_true, cond_number)
            
            if symmetric:
                M_true =  X_true @ X_true.T
            else:
                M_true = X_true @ Y_true.T
                
                
                
            y_true = A(M_true)
    
            radius_x = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
            radius_y = init_radius_ratio*np.linalg.norm(Y_true, ord='fro')


            X_padded = np.zeros((n, r))
            Y_padded = np.zeros((n, r))
            X_padded[:, :r_true] = X_true
            Y_padded[:, :r_true] = Y_true
            
            X0 = gen_random_point_in_neighborhood(X_true, radius_x, r, r_true)
            Y0 = gen_random_point_in_neighborhood(Y_true, radius_y, r, r_true)
            

        
            for method in methods:
                if symmetric:
                    matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial)
                else:
                    matrix_recovery_assymetric(X0, Y0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, method, base_dir, trial)
                    
    return 'we dont care'



def collect_compute_mean(keys, loss_ord, r_true, res, methods):
    losses = dict(( method, dict()) for method in methods )

    for rank, cond_number in keys:
        for method in methods:
            file_pattern = f"experiments/{'res' if res else 'exp'}_{method}_l_{loss_ord}_r*={r_true}_r={rank}_condn={cond_number}_trial_*.csv"
            file_list = glob.glob(file_pattern)
            data_list = []
            
            # Read each file and append its data to the data_list
            for file in file_list:
                data_list.append(np.loadtxt(file, delimiter=','))  # Assume the CSV is correctly formatted for np.loadtxt
            
            # Convert the list of arrays into a 2D numpy array for easier manipulation
            data_array = np.array(data_list)
            
            # Compute the mean across all trials (rows) for each experiment
            mean_values = np.mean(data_array, axis=0)
            losses[method][(rank, cond_number)]  = mean_values
   
    return losses



if __name__ == "__main__":
    
    
    loss_ord = 1
    r_true = 2
    n_cpu = 1
    n_trial_div_n_cpu = 1
    os.system('rm experiments/*.csv')
    T =1000
    n = 20
    np.random.seed(42)
    methods = [ 'gnp', 'subGD', 'scaled_subGD' ]
    init_radius_ratio = 0.1
    keys = [ (2,1),(2,100), (5,1)]
    symmetric= True
    
    d = 10 * n * r_true
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if n_cpu > 1:
        
        processes = [  Process(name=f"cpu {cpu}", target=partial(trial_execution, range(cpu*n_trial_div_n_cpu, (cpu+1)*n_trial_div_n_cpu), n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric))
                    for cpu in range(n_cpu) ]  
    
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
    else:
        trial_execution(range(0, n_trial_div_n_cpu), n, r_true, d, keys, init_radius_ratio, T, loss_ord, base_dir, methods, symmetric)
    

    losses = collect_compute_mean(keys, loss_ord, r_true, False, methods)
    plot_losses_with_styles(losses, r_true, loss_ord, base_dir, symmetric)

    
