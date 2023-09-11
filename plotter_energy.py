import numpy as np
import matplotlib.pyplot as plt
import os
import wass_opt_lib
import time

# USER VARIABLES
# experiment_name = '3_energy'

# experiment_name = '19_7_energy_0.1'
# experiment_name = '19_7_energy_0.2'
# experiment_name = '19_7_energy_0.5'
# 

# BEGIN

# V_name = 'mix_gaussian'
# V_name = 'double_banana' # experiment_name ='21_7_energy_0.1_0.1'
# V_name = 'double_banana2'
V_name = 'long_gaussian'
experiment_name ='21_7_energy_0.1'


data_path = os.path.join("data", V_name, experiment_name)
# save empiricals

params = np.load(os.path.join(data_path,"params.npz"))
beta = params['beta']
maximum_T = params['maximum_T']
init_mu = params['init_mu']
init_var = params['init_var']
stepsize = params['stepsize']
WP_T_arr = params['WP_T_arr']

steps_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)*stepsize

figs_path = os.path.join('figs', V_name, "init_mu"+str(init_mu)+"var"+str(init_var),"beta"+str(beta), "stepsize"+str(stepsize))

# SGD
# if not os.path.exists(os.path.join(data_path,"empirical")):
#     os.makedirs(os.path.join(data_path,"empirical"))

X_SGD = np.load(os.path.join(data_path,"empirical", "SGD.npy"))
energybar_sgd = np.load(os.path.join(data_path,"empirical", "energy_SGD.npy"))


X_MALA = np.load(os.path.join(data_path,"empirical", "MALA.npy"))
energybar_MALA = np.load(os.path.join(data_path,"empirical", "energy_MALA.npy"))


#  X_KER1 = np.load(os.path.join(data_path,"empirical", "WP_KER1"))
X_KER2 = np.load(os.path.join(data_path,"empirical", "WP_KER2.npy"))

energybar_KER2 = np.load(os.path.join(data_path,"empirical", "energy_KER2.npy"))

# PLot the energys

fig_energy_WP, ax_energy_WP = plt.subplots()
fig_energy_WP.suptitle("Empirical KL between rho_T and (rho_true)_T")
ax_energy_WP.plot(steps_arr, energybar_sgd, label="SGD")
ax_energy_WP.plot(steps_arr, energybar_MALA, label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001: #or WP_T == 0.1:
        continue
    ax_energy_WP.plot(steps_arr, energybar_KER2[ctr,:], label="WProx Kernel T="+str(WP_T))

ax_energy_WP.legend()
fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T"))


# ax_energy_WP.set_yscale('log')
# fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T_log"))