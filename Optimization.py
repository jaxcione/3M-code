#%% import header block
import os
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, acquisition
from final import black_box

# make folders for saved figures
os.makedirs('bo_figures/stage1_temperature', exist_ok=True)
os.makedirs('bo_figures/stage2_H_tf_3d', exist_ok=True)

#%% acquisition function
acq = acquisition.UpperConfidenceBound(kappa=10)

#%% helper function: run one BO problem and return full results
def run_bo(pbounds, init_points=3, n_iter=47, random_state=None):
    optimizer = BayesianOptimization(
        f=black_box,
        acquisition_function=acq,
        pbounds=pbounds,
        random_state=random_state,
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    res = optimizer.res

    out = {
        'optimizer': optimizer,
        'res': res,
        'targets': np.array([entry['target'] for entry in res]),
        'T_vals': np.array([entry['params']['T'] for entry in res]),
        'H_vals': np.array([entry['params']['H'] for entry in res]),
        'tf_vals': np.array([entry['params']['t_f'] for entry in res]),
        'best_target': optimizer.max['target'],
        'best_T': optimizer.max['params']['T'],
        'best_H': optimizer.max['params']['H'],
        'best_tf': optimizer.max['params']['t_f'],
    }
    return out


#%% stage 1: optimize temperature for several fixed (H, t_f) combinations
H_opt = 0.010884
tf_opt = 6719.3420

H_grid = [H_opt - 0.005, H_opt, H_opt + 0.005]
tf_grid = [tf_opt - 150, tf_opt, tf_opt + 150]

temp_runs = []
best_T_list = []
best_target_list_stage1 = []

for H_fixed in H_grid:
    for tf_fixed in tf_grid:
        pbounds = {
            'T': (255, 310),
            'H': (H_fixed, H_fixed),
            't_f': (tf_fixed, tf_fixed),
        }

        run_data = run_bo(
            pbounds=pbounds,
            init_points=3,
            n_iter=97,
            random_state=None,
        )

        run_data['H_fixed'] = H_fixed
        run_data['tf_fixed'] = tf_fixed

        temp_runs.append(run_data)
        best_T_list.append(run_data['best_T'])
        best_target_list_stage1.append(run_data['best_target'])

best_T_array = np.array(best_T_list)
T_avg = np.mean(best_T_array)
T_var = np.var(best_T_array, ddof=1)
T_std = np.std(best_T_array, ddof=1)

print('\nStage 1: temperature optimization summary')
print(f'Average best T = {T_avg:.4f} K')
print(f'Variance of best T = {T_var:.6f}')
print(f'Std of best T = {T_std:.6f}')


#%% stage 2: for several fixed T values, optimize H and t_f
T_avg = 286.4494
T_test_values = [T_avg - 5, T_avg - 2.5, T_avg, T_avg + 2.5, T_avg + 5]

Ht_runs = []
best_H_list = []
best_tf_list = []
best_target_list_stage2 = []

for T_fixed in T_test_values:
    for rep in range(3):
        pbounds = {
            'T': (T_fixed, T_fixed),
            'H': (0.01, 0.25),
            't_f': (300, 7200),
        }

        run_data = run_bo(
            pbounds=pbounds,
            init_points=3,
            n_iter=97,
            random_state=None,
        )

        run_data['T_fixed'] = T_fixed
        run_data['rep'] = rep + 1

        Ht_runs.append(run_data)
        best_H_list.append(run_data['best_H'])
        best_tf_list.append(run_data['best_tf'])
        best_target_list_stage2.append(run_data['best_target'])

best_H_array = np.array(best_H_list)
best_tf_array = np.array(best_tf_list)

H_avg = np.mean(best_H_array)
H_var = np.var(best_H_array, ddof=1)
H_std = np.std(best_H_array, ddof=1)

tf_avg = np.mean(best_tf_array)
tf_var = np.var(best_tf_array, ddof=1)
tf_std = np.std(best_tf_array, ddof=1)

print('\nStage 2: H and t_f optimization summary')
print(f'Average best H = {H_avg:.6f} m')
print(f'Variance of best H = {H_var:.6e}')
print(f'Std of best H = {H_std:.6e}')
print(f'Average best t_f = {tf_avg:.4f} s')
print(f'Variance of best t_f = {tf_var:.6e}')
print(f'Std of best t_f = {tf_std:.6e}')


#%% plot helper: temperature vs target for a chosen stage-1 run
def plot_temp_run(run_data, run_number=None, save_dir='bo_figures/stage1_temperature'):
    T_vals = run_data['T_vals']
    targets = run_data['targets']

    sort_idx = np.argsort(T_vals)
    T_sorted = T_vals[sort_idx]
    targets_sorted = targets[sort_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(T_vals, targets, label='BO samples')
    plt.plot(T_sorted, targets_sorted, lw=1.5, label='Sampled trend')
    plt.xlabel('T [K]')
    plt.ylabel('Net CO$_2$ captured')
    plt.title(
        f'Target vs Temperature | H = {run_data["H_fixed"]:.3f} m, '
        f't_f = {run_data["tf_fixed"]:.0f} s'
    )
    plt.legend()
    plt.grid()

    if run_number is None:
        filename = (
            f'{save_dir}/temp_optimization_'
            f'H_{run_data["H_fixed"]:.3f}_tf_{run_data["tf_fixed"]:.0f}.png'
        )
    else:
        filename = (
            f'{save_dir}/temp_optimization_run_{run_number:02d}_'
            f'H_{run_data["H_fixed"]:.3f}_tf_{run_data["tf_fixed"]:.0f}.png'
        )

    plt.savefig(filename, bbox_inches='tight')
    plt.show()


#%% plot several temperature runs
example_indices = [0, 5, 8]
for idx in example_indices:
    plot_temp_run(temp_runs[idx], run_number=idx)


#%% plot helper: 3D scatter for a chosen stage-2 run
def plot_H_tf_run_3d(run_data, run_number=None, save_dir='bo_figures/stage2_H_tf_3d'):
    H_vals = run_data['H_vals']
    tf_vals = run_data['tf_vals']
    targets = run_data['targets']

    best_H = run_data['best_H']
    best_tf = run_data['best_tf']
    best_target = run_data['best_target']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(H_vals, tf_vals, targets, label='BO samples')
    ax.scatter(best_H, best_tf, best_target, s=80, label='Best point')

    ax.set_xlabel('H [m]')
    ax.set_ylabel('t_f [s]')
    ax.set_zlabel('Objective')
    ax.set_zlim(0, 5)

    ax.set_title(
        f'Objective vs H and t_f | T = {run_data["T_fixed"]:.2f} K, '
        f'run {run_data["rep"]}'
    )
    ax.legend()

    if run_number is None:
        filename = (
            f'{save_dir}/Ht_optimization_'
            f'T_{run_data["T_fixed"]:.2f}_rep_{run_data["rep"]}.png'
        )
    else:
        filename = (
            f'{save_dir}/Ht_optimization_run_{run_number:02d}_'
            f'T_{run_data["T_fixed"]:.2f}_rep_{run_data["rep"]}.png'
        )

    plt.savefig(filename, bbox_inches='tight')
    plt.show()


#%% plot several 3D H-t_f runs
example_stage2_indices = [0, 5, 14]
for idx in example_stage2_indices:
    plot_H_tf_run_3d(Ht_runs[idx], run_number=idx)


#%% save all results
np.savez_compressed(
    'bayes_opt_summary.npz',
    best_T_array=best_T_array,
    best_H_array=best_H_array,
    best_tf_array=best_tf_array,
    T_avg=T_avg,
    T_var=T_var,
    T_std=T_std,
    H_avg=H_avg,
    H_var=H_var,
    H_std=H_std,
    tf_avg=tf_avg,
    tf_var=tf_var,
    tf_std=tf_std,
)

print('\nSaved summary statistics to bayes_opt_summary.npz')