import numpy as np
import matplotlib.pyplot as plt
import os
import wass_opt_lib
import time

# USER VARIABLES


# BEGIN


problem_dim = 2
logistic_samples=50
# problem_dim = 50
# logistic_samples=500
beta=1.0
# stepsize=0.0005
stepsize=0.05
# alpha=0.1
alpha=0.5
# stepsize=0.05

data_path = os.path.join("data", "logreg_TM", str(problem_dim)+'_'+str(logistic_samples)+'_'+str(alpha),"beta"+str(beta)+"ss"+str(stepsize))
# save empiricals

params = np.load(os.path.join(data_path,"params.npz"))
beta = params['beta']
stepsize = params['stepsize']
WP_T_arr = params['WP_T_arr']
n_iters = params['n_iters']
steps_arr = np.arange(0, n_iters+1)

figs_path = os.path.join('figs', "logreg_tm", str(problem_dim)+'_'+str(logistic_samples)+'_'+str(alpha),"beta"+str(beta)+"ss"+str(stepsize))

# SGD
# if not os.path.exists(os.path.join(data_path,"empirical")):
#     os.makedirs(os.path.join(data_path,"empirical"))

X_SGD = np.load(os.path.join(data_path,"empirical", "SGD.npy"))
errbar_sgd = np.load(os.path.join(data_path,"empirical", "err_SGD.npy"))
err2bar_sgd = np.load(os.path.join(data_path,"empirical", "err2_SGD.npy"))

X_MALA = np.load(os.path.join(data_path,"empirical", "MALA.npy"))
errbar_MALA = np.load(os.path.join(data_path,"empirical", "err_MALA.npy"))
err2bar_MALA = np.load(os.path.join(data_path,"empirical", "err2_MALA.npy"))

#  X_KER1 = np.load(os.path.join(data_path,"empirical", "WP_KER1"))
X_KER2 = np.load(os.path.join(data_path,"empirical", "WP_KER2.npy"))
errbar_KER2 = np.load(os.path.join(data_path,"empirical", "err_KER2.npy"))
err2bar_KER2 = np.load(os.path.join(data_path,"empirical", "err2_KER2.npy"))
# PLot the errs


fig_err_WP, ax_err_WP = plt.subplots()
# fig_err_WP.suptitle("Error")
ax_err_WP.plot(steps_arr, errbar_sgd, label="ULA")
ax_err_WP.plot(steps_arr, errbar_MALA, label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001: #or WP_T == 0.1:
        continue
    ax_err_WP.plot(steps_arr, errbar_KER2[ctr,:], label="BRWP T="+str(WP_T))

ax_err_WP.set_xlabel("Iteration")
ax_err_WP.set_ylabel(r"$\mathbb{E}||\theta - \theta^*||_1/d$")
ax_err_WP.legend()
fig_err_WP.savefig(os.path.join(figs_path, "std_empirical_T"))


ax_err_WP.set_yscale('log')
fig_err_WP.savefig(os.path.join(figs_path, "std_empirical_T_log"))

fig_err_WP, ax_err_WP = plt.subplots()
# fig_err_WP.suptitle("Error")
ax_err_WP.plot(steps_arr, err2bar_sgd, label="ULA")
ax_err_WP.plot(steps_arr, err2bar_MALA, label="MALA")
for ctr, WP_T in enumerate(WP_T_arr):
    if WP_T == 0.001: #or WP_T == 0.1:
        continue
    ax_err_WP.plot(steps_arr, err2bar_KER2[ctr,:], label="BRWP T="+str(WP_T))

ax_err_WP.set_xlabel("Iteration")
ax_err_WP.set_ylabel(r"$||\bar{\theta} - \theta^*||_1/d$")
ax_err_WP.legend()
fig_err_WP.savefig(os.path.join(figs_path, "err_empirical_T"))

ax_err_WP.set_yscale('log')
fig_err_WP.savefig(os.path.join(figs_path, "err_empirical_T_log"))