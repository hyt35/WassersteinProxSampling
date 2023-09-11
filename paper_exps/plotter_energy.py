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

V_name = 'mix_gaussian'
experiment_name ='21_7_energy_0.1'

# V_name = 'double_banana2'
# V_name = 'long_gaussian'
# experiment_name = '7_8_energy_0.1'
# experiment_name = '7_8_energy_0.01'


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
ax_energy_WP.set_xlabel("Time")
# fig_energy_WP.suptitle("Empirical KL between rho_T and (rho_true)_T")
ax_energy_WP.plot(steps_arr, energybar_sgd, label="ULA")
ax_energy_WP.plot(steps_arr, energybar_MALA, label="MALA")
# ax_energy_WP.plot(steps_arr[:50], energybar_sgd[:50], label="ULA")
# ax_energy_WP.plot(steps_arr[:50], energybar_MALA[:50], label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001: #or WP_T == 0.1:
        continue
    ax_energy_WP.plot(steps_arr, energybar_KER2[ctr,:], label="BRWP T="+str(WP_T))
    # ax_energy_WP.plot(steps_arr[:50], energybar_KER2[ctr,:50], label="BRWP T="+str(WP_T))

ax_energy_WP.legend()
fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T"))

# a = np.diag(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
# sigma = np.linalg.inv(a)
# var_t = sigma * (1-0.01**2 * np.linalg.inv(sigma)**2)
# # print(np.cov(X_SGD.T).shape)
# print(np.linalg.norm(np.linalg.inv(np.cov(X_SGD.T)) - np.linalg.inv(sigma), ord='fro'))
# print(np.linalg.norm(np.linalg.inv(np.cov(X_MALA.T)) - np.linalg.inv(sigma), ord='fro'))
# for ctr, WP_T in enumerate(WP_T_arr):
#     var_t = sigma * (1-WP_T**2 * np.linalg.inv(sigma)**2)
#     print(np.linalg.norm(np.linalg.inv(np.cov(X_KER2.T)) - np.linalg.inv(sigma), ord='fro'))
#     print(np.linalg.norm(np.linalg.inv(np.cov(X_KER2.T)) - np.linalg.inv(var_t), ord='fro'))
# print(np.cov(X_KER2.T))
# print(np.cov(X_SGD))
# print(np.cov(X_SGD))
# ax_energy_WP.set_yscale('log')
# fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T_log"))