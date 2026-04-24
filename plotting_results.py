#%% import header block
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200
})

# make folder for figures
os.makedirs('bo_summary_figures', exist_ok=True)


#%% load saved optimization summary data
data = np.load('bayes_opt_summary.npz')

best_T_array = data['best_T_array']
best_H_array = data['best_H_array']
best_tf_array = data['best_tf_array']

T_avg = data['T_avg'].item()
T_var = data['T_var'].item()
T_std = data['T_std'].item()

H_avg = data['H_avg'].item()
H_var = data['H_var'].item()
H_std = data['H_std'].item()

tf_avg = data['tf_avg'].item()
tf_var = data['tf_var'].item()
tf_std = data['tf_std'].item()

print('\nLoaded summary statistics from bayes_opt_summary.npz')
print(f'Average best T   = {T_avg:.4f} K')
print(f'Variance best T  = {T_var:.6e}')
print(f'Std best T       = {T_std:.6e}')
print(f'Average best H   = {H_avg:.6f} m')
print(f'Variance best H  = {H_var:.6e}')
print(f'Std best H       = {H_std:.6e}')
print(f'Average best t_f = {tf_avg:.4f} s')
print(f'Variance best t_f= {tf_var:.6e}')
print(f'Std best t_f     = {tf_std:.6e}')


#%% run indices
runs_T = np.arange(1, len(best_T_array) + 1)
runs_Htf = np.arange(1, len(best_H_array) + 1)


#%% best temperature by run
plt.figure(figsize=(8, 4))
plt.plot(runs_T, best_T_array, 'o-', ms=4, label='Best T from each run')
plt.axhline(T_avg, lw=2, linestyle='--', label=f'Average = {T_avg:.2f} K')

plt.xlabel('Run index')
plt.ylabel('Best T [K]')
plt.title('Best temperature from Stage 1 runs')
plt.grid(alpha=0.3)
plt.legend()

plt.savefig('bo_summary_figures/best_temperature_by_run.png', bbox_inches='tight')
plt.show()


#%% histogram of best temperatures
plt.figure(figsize=(7, 4))
plt.hist(best_T_array, bins=10, edgecolor='black', alpha=0.75)
plt.axvline(T_avg, lw=2, linestyle='--', label=f'Mean = {T_avg:.2f} K')

plt.xlabel('Best T [K]')
plt.ylabel('Count')
plt.title('Distribution of best temperature values')
plt.grid(alpha=0.3)
plt.legend()

plt.savefig('bo_summary_figures/best_temperature_histogram.png', bbox_inches='tight')
plt.show()


#%% best H and best t_f by run
fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

ax[0].plot(runs_Htf, best_H_array, 'o-', ms=4, label='Best H from each run')
ax[0].axhline(H_avg, lw=2, linestyle='--', label=f'Average = {H_avg:.4f} m')
ax[0].set_ylabel('Best H [m]')
ax[0].set_title('Best H values from Stage 2 runs')
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].plot(runs_Htf, best_tf_array, 'o-', ms=4, label='Best t_f from each run')
ax[1].axhline(tf_avg, lw=2, linestyle='--', label=f'Average = {tf_avg:.1f} s')
ax[1].set_ylabel(r'Best $t_f$ [s]')
ax[1].set_xlabel('Run index')
ax[1].set_title('Best t_f values from Stage 2 runs')
ax[1].grid(alpha=0.3)
ax[1].legend()

plt.savefig('bo_summary_figures/best_H_and_tf_by_run.png', bbox_inches='tight')
plt.show()


#%% histograms of best H and best t_f
fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

ax[0].hist(best_H_array, bins=5, edgecolor='black', alpha=0.75)
ax[0].axvline(H_avg, lw=2, linestyle='--', label=f'Mean = {H_avg:.4f} m')
ax[0].set_xlabel('Best H [m]')
ax[0].set_ylabel('Count')
ax[0].set_title('Distribution of best H values')
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].hist(best_tf_array, bins=5, edgecolor='black', alpha=0.75)
ax[1].axvline(tf_avg, lw=2, linestyle='--', label=f'Mean = {tf_avg:.1f} s')
ax[1].set_xlabel(r'Best $t_f$ [s]')
ax[1].set_ylabel('Count')
ax[1].set_title('Distribution of best t_f values')
ax[1].grid(alpha=0.3)
ax[1].legend()

plt.savefig('bo_summary_figures/best_H_and_tf_histograms.png', bbox_inches='tight')
plt.show()


#%% pairwise scatter of best values from stage 2
plt.figure(figsize=(7, 5))
plt.scatter(best_H_array, best_tf_array, s=45, alpha=0.75)
plt.scatter(H_avg, tf_avg, marker='*', s=220, label='Average point')

plt.xlabel('Best H [m]')
plt.ylabel(r'Best $t_f$ [s]')
plt.title('Best H and t_f pairs from Stage 2 runs')
plt.grid(alpha=0.3)
plt.legend()

plt.savefig('bo_summary_figures/best_H_tf_scatter.png', bbox_inches='tight')
plt.show()