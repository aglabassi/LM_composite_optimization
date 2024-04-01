import sys,os
from functools import partial
from multiprocessing import Process
import numpy as np
import glob
from utils import create_rip_transform, generate_matrix_with_condition, gen_random_point_in_neighborhood, matrix_recovery, plot_losses_with_styles

def trial_execution(trials, n, r_true, d, cond_numbers, ranks, init_radius_ratio, T, loss_ord, lambdaa_scaled, lambdaa_gnp, base_dir):
    
    for trial in trials:
        A, A_adj = create_rip_transform(n, d)
        losses_scaled_trial = []
        losses_gnp_trial = []
    
        for cond_number in cond_numbers:
            X_true = generate_matrix_with_condition(n, r_true, cond_number)
            M_true = X_true @ X_true.T
            y_true = A(M_true)
    
            radius = init_radius_ratio*np.linalg.norm(X_true, ord='fro')
    
            for r in ranks:
                X0 = gen_random_point_in_neighborhood(X_true, radius, r, r_true)
                losses_scaled_trial.append(matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, lambdaa_scaled, 'scaled', base_dir, trial))
                losses_gnp_trial.append(matrix_recovery(X0, M_true, T, A, A_adj, y_true, loss_ord, r_true, cond_number, lambdaa_gnp, 'gnp', base_dir, trial))

    return 'we dont care'



def collect_compute_mean(ranks, cond_numbers, loss_ord, r_true, res):
    losses_scaled = []
    losses_gnp = []
    for cond_number in cond_numbers:
        for rank in ranks:
            for method in ['scaled', 'gnp']:
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
                if method == 'scaled':
                    losses_scaled.append(mean_values)
                else:
                    losses_gnp.append(mean_values)
    
    return losses_scaled, losses_gnp




if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        try:
            loss_ord = int(sys.argv[1])  # Convert first command-line argument to integer
        except ValueError:
            print("Please provide an integer value for loss_ord.")
            sys.exit(1)
    else:
        loss_ord = 2  # Default value if not provided
        print(f"No loss_ord provided, using default value of {loss_ord}.")

        
 
    r_true = 3
    n_cpu = 1
    n_trial_div_n_cpu = 1
    
    T = 1000
    n = 100
    np.random.seed(42)
    lambdaa_gnp  = 'Liwei'
    lambdaa_scaled = 'Liwei'
    init_radius_ratio = 0.01
    ranks_test = [3,20,99]
    cond_numbers_test = [1,1000]
    
    d = 10 * n * r_true
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if n_cpu > 1:
        
        processes = [  Process(name=f"cpu {cpu}", target=partial(trial_execution, range(cpu*n_trial_div_n_cpu, (cpu+1)*n_trial_div_n_cpu), n, r_true, d, cond_numbers_test, ranks_test, init_radius_ratio, T, loss_ord, lambdaa_scaled, lambdaa_gnp, base_dir))
                    for cpu in range(n_cpu) ]  
    
        
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
    else:
        trial_execution(range(0, n_trial_div_n_cpu), n, r_true, d, cond_numbers_test, ranks_test, init_radius_ratio, T, loss_ord, lambdaa_scaled, lambdaa_gnp, base_dir)
        
        
    losses_scaled, losses_gnp = collect_compute_mean(ranks_test, cond_numbers_test, loss_ord, r_true, False)
    res_scaled, res_gnp = collect_compute_mean(ranks_test, cond_numbers_test, loss_ord, r_true, True)
    
    plot_losses_with_styles(losses_scaled, losses_gnp, lambdaa_scaled, lambdaa_gnp, cond_numbers_test, ranks_test, r_true, loss_ord, base_dir, n_trial_div_n_cpu*n_cpu, False)
    plot_losses_with_styles(res_scaled, res_gnp, lambdaa_scaled, lambdaa_gnp, cond_numbers_test, ranks_test, r_true, loss_ord, base_dir, n_trial_div_n_cpu*n_cpu, True)

    
    
