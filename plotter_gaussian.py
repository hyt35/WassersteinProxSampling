import numpy as np
import matplotlib.pyplot as plt
import os
import wass_opt_lib
import time

# USER VARIABLES
# experiment_name = '3_energy'

# experiment_name = '19_7_energy_0.1'
# experiment_name = '19_7_energy_0.2'
experiment_name = '19_7_energy_0.5'
# 

# BEGIN



data_path = os.path.join("data", "gaussian_exps", experiment_name)
# save empiricals

params = np.load(os.path.join(data_path,"params.npz"))
a = params['a']
beta = params['beta']
maximum_T = params['maximum_T']
init_mu = params['init_mu']
init_var = params['init_var']
stepsize = params['stepsize']
WP_T_arr = params['WP_T_arr']

steps_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)*stepsize

figs_path = os.path.join('figs', 'gaussian_exps', "mu"+str(init_mu)+"var"+str(init_var), "a"+str(a)+"beta"+str(beta), "stepsize"+str(stepsize))

target_mu = 0
target_var = beta/a

# SGD
# if not os.path.exists(os.path.join(data_path,"empirical")):
#     os.makedirs(os.path.join(data_path,"empirical"))

X_SGD = np.load(os.path.join(data_path,"empirical", "SGD.npy"))
energybar_sgd = np.load(os.path.join(data_path,"empirical", "energy_SGD.npy"))
energybar0_sgd = np.load(os.path.join(data_path,"empirical", "energy0_SGD.npy"))

X_MALA = np.load(os.path.join(data_path,"empirical", "MALA.npy"))
energybar_MALA = np.load(os.path.join(data_path,"empirical", "energy_MALA.npy"))
energybar0_MALA = np.load(os.path.join(data_path,"empirical", "energy0_MALA.npy"))

#  X_KER1 = np.load(os.path.join(data_path,"empirical", "WP_KER1"))
X_KER2 = np.load(os.path.join(data_path,"empirical", "WP_KER2.npy"))
X_KER3 = np.load(os.path.join(data_path,"empirical", "WP_KER3.npy"))
energybar_KER2 = np.load(os.path.join(data_path,"empirical", "energy_KER2.npy"))
energybar_KER3 = np.load(os.path.join(data_path,"empirical", "energy_KER3.npy"))
energybar0_KER2 = np.load(os.path.join(data_path,"empirical", "energy0_KER2.npy"))
energybar0_KER3 = np.load(os.path.join(data_path,"empirical", "energy0_KER3.npy"))
# PLot the energys

fig_energy_WP, ax_energy_WP = plt.subplots()
fig_energy_WP.suptitle("Empirical KL between rho_T and (rho_true)_T")
ax_energy_WP.plot(steps_arr, energybar_sgd, label="SGD")
ax_energy_WP.plot(steps_arr, energybar_MALA, label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001: #or WP_T == 0.1:
        continue
    ax_energy_WP.plot(steps_arr, energybar_KER2[ctr,:], label="WProx Kernel T="+str(WP_T))
    ax_energy_WP.plot(steps_arr, energybar_KER3[ctr,:], label="WProx Gaussian T="+str(WP_T))
ax_energy_WP.legend()
fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T"))

fig_energy0_WP, ax_energy0_WP = plt.subplots()
fig_energy0_WP.suptitle("KL between rho_T and (rho_true)")
ax_energy0_WP.plot(steps_arr, energybar0_sgd, label="SGD")
ax_energy0_WP.plot(steps_arr, energybar0_MALA, label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001 :#or WP_T == 0.1:
        continue
    ax_energy0_WP.plot(steps_arr, energybar0_KER2[ctr,:], label="WProx Kernel T="+str(WP_T))
    ax_energy0_WP.plot(steps_arr, energybar0_KER3[ctr,:], label="WProx Gaussian T="+str(WP_T))
ax_energy0_WP.legend()
fig_energy0_WP.savefig(os.path.join(figs_path, "energy0_empirical_T"))

ax_energy_WP.set_yscale('log')
fig_energy_WP.savefig(os.path.join(figs_path, "energy_empirical_T_log"))

ax_energy0_WP.set_yscale('log')
fig_energy0_WP.savefig(os.path.join(figs_path, "energy0_empirical_T_log"))


'''
For computing the difference between the empirical and the true KL distance for the
'''
def KL_twogaussians(param1, param2, T1=0, T2=0):
    mean1, var1 = param1
    mean2, var2 = param2
    mean1_T = mean1/(1+a*T1)
    mean2_T = mean2/(1+a*T2)

    var1_T = var1/((1+a*T1)**2) + (2*beta*T1)/(1+a*T1)
    var2_T = var2/((1+a*T2)**2) + (2*beta*T2)/(1+a*T2)


    # compute 
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    energy = np.log(var2_T/var1_T)/2 + (var1_T + (mean1_T-mean2_T)**2)/(2*var2_T) - 0.5
    return energy
index_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)
steps_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)*stepsize
true_mu = init_mu * np.exp(-a*steps_arr)
true_var = init_var * np.exp(-2*a*steps_arr) + beta*(1-np.exp(-2*a*steps_arr))/a
'''
SGD
'''
# create grid
start = time.time()

# grid_X_SGD = steps_arr*stepsize

# mu_{k} = (1-a*eta)^k
mu_SGD = np.power(1-a*stepsize, index_arr)
var_SGD = np.power(1-a*stepsize, index_arr*2)*init_var + 2*beta*stepsize*(np.cumsum(np.power(1-a*stepsize, index_arr*2-2)) - np.power(1-a*stepsize, -2))
energy_SGDtrue = KL_twogaussians((mu_SGD, var_SGD), (target_mu, target_var), 0,0)

print("SGD analytic", time.time()-start)
#%%
'''
Wasserstein Prox
'''
start = time.time()

beta = beta
mu_WP_arr = np.empty((len(WP_T_arr), len(steps_arr)))
var_WP_arr = np.empty((len(WP_T_arr), len(steps_arr)))
energy_WP_arr = np.empty((len(WP_T_arr), len(steps_arr)))
energy_WP_arr_true = np.empty((len(WP_T_arr), len(steps_arr)))
for ctri, WP_T in enumerate(WP_T_arr):
    # grid_X_WProx = steps_arr*stepsize
    ctr = 0
    mu_WP = np.empty(len(steps_arr))
    var_WP = np.empty(len(steps_arr))

    for ctr in range(len(steps_arr)):
        if ctr == 0:
            mu = init_mu
            var = init_var
        else:
            mu = (1-a*stepsize + (stepsize*beta*(1+a*WP_T)*a*WP_T)/(var + 2*beta*WP_T*(1+a*WP_T)))*mu
            var = (1-a*stepsize + (stepsize*beta*(1+a*WP_T)**2)/(var + 2*beta*WP_T*(1+a*WP_T)))**2*var
        mu_WP[ctr] = mu
        var_WP[ctr] = var

    energy_WP_arr[ctri] = KL_twogaussians((mu_WP, var_WP), (true_mu, true_var), 0, 0)
    energy_WP_arr_true[ctri] = KL_twogaussians((mu_WP, var_WP), (target_mu, target_var), 0, 0)

# KL(rho(t), pi) where rho evolves as Fokker planck
energy_FokkerPlanck = KL_twogaussians((true_mu, true_var), (target_mu, target_var),0,0)





fig_WP_difftotrue, ax_WP_difftotrue = plt.subplots()
fig_WP_difftotrue.suptitle("KL(rho_T^approx(t), pi) - KL(rho^true(t), pi)")

ax_WP_difftotrue.plot(steps_arr, energybar0_sgd - energy_FokkerPlanck, label="ULA")
ax_WP_difftotrue.plot(steps_arr, energybar0_MALA - energy_FokkerPlanck, label="MALA")
for ctri, WP_T in enumerate(WP_T_arr):
    ax_WP_difftotrue.plot(steps_arr, energybar0_KER2[ctri,:] - energy_FokkerPlanck, label="WProx Kernel T="+str(WP_T))
    ax_WP_difftotrue.plot(steps_arr, energybar0_KER3[ctri,:] - energy_FokkerPlanck, label="WProx Gaussian T="+str(WP_T))

ax_WP_difftotrue.legend()
fig_WP_difftotrue.savefig(os.path.join(figs_path, "energy_difftotrue"))


fig_WP_vs_analytic, ax_WP_vs_analytic = plt.subplots()
fig_WP_vs_analytic.suptitle("KL discretization error for WProx")
for ctri, WP_T in enumerate(WP_T_arr):
    ax_WP_vs_analytic.plot(steps_arr, energybar0_KER2[ctri,:] - energy_WP_arr_true[ctri,:], label="WProx Kernel T="+str(WP_T))
    ax_WP_vs_analytic.plot(steps_arr, energybar0_KER3[ctri,:] - energy_WP_arr_true[ctri,:], label="WProx Gaussian T="+str(WP_T))
ax_WP_vs_analytic.legend()
fig_WP_vs_analytic.savefig(os.path.join(figs_path, "energy_vs_analytic"))