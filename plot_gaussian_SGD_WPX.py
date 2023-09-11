import numpy as np
import matplotlib.pyplot as plt
import os
import wass_opt_lib
import time

# USER VARIABLES
a = 1
beta = 1
maximum_T = 10
init_mu = 1
init_var = 4

WP_T_arr = [0.1, 0.5, 1.0, 2.0]
experiment_name = '19_7_energy_0.2'
stepsize = 0.2
# experiment_name = '19_7_energy_0.1'
# stepsize = 0.1

# experiment_name = '19_7_energy_0.5'
# stepsize = 0.5

SGD_ON = True
EMPIRICAL_ON = False
MALA_ON = True
WP_ON = True
ENERGY_ON = True
# BEGIN
figs_path = os.path.join('figs', 'gaussian_exps', "mu"+str(init_mu)+"var"+str(init_var), "a"+str(a)+"beta"+str(beta), "stepsize"+str(stepsize))

target_mu = 0
target_var = beta/a

if not os.path.exists(figs_path):
    os.makedirs(figs_path)

def f(x):
    return a*(x**2)/2

def df(x):
    return a*x

# def compute_KL(x):
#     # computes the KL divergence between samples x and the target distribution
#     t_mu = target_mu
#     t_var = target_var
#     if len(x.shape) == 1:
#         x = x[:,None]
    
def compute_KL(x, T=0.01, T_target=0.01):
    # computes the KL divergence between samples x and WProx of target distribution
    t_mu = target_mu
    t_var = target_var/((1+a*T_target)**2) + (2*beta*T_target)/(1+a*T_target)
    if len(x.shape) == 1:
        x = x[:,None]
    if T<= 0 or T_target<0:
        raise Exception("T should be nonnegative")

    def V(x):
        foo = (x-t_mu)**2/(2*t_var)
        return foo[:,0]
    # def dV(x):
    #     return a*(x-t_mu)/t_var

    gaussian_normalizing_const = np.sqrt(2*np.pi*t_var)
    energy = wass_opt_lib.compute_energy_standalone_corrected(x, V, beta, T, sample_iters=25)
    # energy = energy/len(x) - np.log(len(x)) + np.log(gaussian_normalizing_const)
    energy = energy+ np.log(gaussian_normalizing_const)
    return energy
    # compute

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

# def KL(x):

grid_X = np.linspace(0, maximum_T, 201, endpoint=True)

true_mu_base = init_mu * np.exp(-a*grid_X)
true_var_base = init_var * np.exp(-2*a*grid_X) + beta*(1-np.exp(-2*a*grid_X))/a
index_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)
steps_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)*stepsize


fig_mu, ax_mu = plt.subplots()
fig_var, ax_var = plt.subplots()

# fig_mu.suptitle("Mean, a="+str(a)+", beta="+str(beta)+", stepsize="+str(stepsize))
# fig_var.suptitle("Variance, a="+str(a)+", beta="+str(beta)+", stepsize="+str(stepsize))

# ax_mu.axhline(y=target_mu)
# ax_var.axhline(y=target_var)

ax_mu.plot(grid_X, true_mu_base, label = "True")
ax_var.plot(grid_X, true_var_base, label = "True")

true_mu = init_mu * np.exp(-a*steps_arr)
true_var = init_var * np.exp(-2*a*steps_arr) + beta*(1-np.exp(-2*a*steps_arr))/a


# Compute true KL divergence

# grid_X_method =steps_arr*stepsize
#%%
'''
SGD
'''
# create grid
start = time.time()

# grid_X_SGD = steps_arr*stepsize

# mu_{k} = (1-a*eta)^k
mu_SGD = np.power(1-a*stepsize, index_arr)
# sigma_k^2 = (1-a*eta)^{2k} sigma_0^2 + 2*beta*eta sum_{j=0}^{k-1} (1-a*eta)^{2j}
var_SGD = np.power(1-a*stepsize, index_arr*2)*init_var + 2*beta*stepsize*(np.cumsum(np.power(1-a*stepsize, index_arr*2-2)) - np.power(1-a*stepsize, -2))
# print(np.cumsum(np.power(1-a*stepsize, steps_arr*2-2)) - np.power(1-a*stepsize, -2))
# print(np.cumsum(np.power(1-a*stepsize, steps_arr*2)))
ax_mu.plot(steps_arr, mu_SGD, label = "ULA")
ax_var.plot(steps_arr, var_SGD,  label = "ULA")

energy_SGD = KL_twogaussians((mu_SGD, var_SGD), (true_mu, true_var), 0, 0)
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
    mu_WP_arr[ctri, :] = mu_WP
    var_WP_arr[ctri, :] = var_WP
    ax_mu.plot(steps_arr, mu_WP, label = "BRWP T="+str(WP_T))
    ax_var.plot(steps_arr, var_WP,  label = "BRWP T="+str(WP_T))




print("WProx analytic", time.time()-start)
#%% 
# Plot
ax_mu.set_xlabel('Time')
ax_var.set_xlabel('Time')
ax_mu.set_ylabel('Mean')
ax_var.set_ylabel('Variance')

ax_mu.legend()
fig_mu.savefig(os.path.join(figs_path, "mu"))

ax_var.legend()
fig_var.savefig(os.path.join(figs_path, "var"))

ax_mu.set_yscale('log')
ax_var.set_yscale('log')
fig_mu.savefig(os.path.join(figs_path, "mu_log"))
fig_var.savefig(os.path.join(figs_path, "var_log"))

ax_mu.set_yscale('linear')
ax_var.set_yscale('linear')

fig_energy, ax_energy = plt.subplots()
fig_energy.suptitle("KL(rho_T(t) || pi_T)")
ax_energy.plot(steps_arr, energy_SGD, label="ULA, T=0")
for ctri, WP_T in enumerate(WP_T_arr):
    ax_energy.plot(steps_arr, energy_WP_arr[ctri], label="WP, T="+str(WP_T))

ax_energy.legend()
fig_energy.savefig(os.path.join(figs_path, "energy"))


fig_energy_true, ax_energy_true = plt.subplots()
fig_energy_true.suptitle("KL(rho_T(t) || pi_infty)")
ax_energy_true.plot(steps_arr, energy_SGDtrue, label="ULA, T=0")
for ctri, WP_T in enumerate(WP_T_arr):
    ax_energy_true.plot(steps_arr, energy_WP_arr_true[ctri], label="WP, T="+str(WP_T))

ax_energy_true.legend()
fig_energy_true.savefig(os.path.join(figs_path, "energy_stationary"))



#%%
'''
Empirical
'''
if EMPIRICAL_ON:
    n_samples = 2000

    X_inits = np.random.randn(n_samples) * np.sqrt(init_var) + init_mu
    init_energy = compute_KL(X_inits)
    print("init energy", init_energy)
    init_energy0 = compute_KL(X_inits, T_target=0)
    print("init energy0", init_energy0)
    '''
    SGD
    '''
    if SGD_ON:
        start = time.time()

        energybar_sgd = np.empty(len(steps_arr))
        mubar_sgd = np.empty(len(steps_arr))
        varbar_sgd = np.empty(len(steps_arr))
        X_SGD = X_inits

        mubar_sgd[0] = np.mean(X_SGD)
        varbar_sgd[0] = np.var(X_SGD)
        energybar_sgd[0] = init_energy

        energybar0_sgd = np.empty(len(steps_arr))
        energybar0_sgd[0] = init_energy0
        for i in range(1,len(steps_arr)):

            X_SGD = (1-a*stepsize) * X_SGD + np.sqrt(2*beta*stepsize) * np.random.randn(n_samples)
            mubar_sgd[i] = np.mean(X_SGD)
            varbar_sgd[i] = np.var(X_SGD) # Biased estimate. 
            energybar_sgd[i] = compute_KL(X_SGD)
            energybar0_sgd[i] = compute_KL(X_SGD, T_target=0)
        print("SGD empirical", time.time()-start)
        print(energybar0_sgd[-1], energybar_sgd[-1])
    #%%
    '''
    MALA
    '''
    if MALA_ON:
        start = time.time()

        if beta != 1:
            print("beta not 1, care")
        mubar_MALA = np.empty(len(steps_arr))
        varbar_MALA = np.empty(len(steps_arr))
        X_MALA = X_inits

        mubar_MALA[0] = np.mean(X_MALA)
        varbar_MALA[0] = np.var(X_MALA)
        energybar_MALA = np.empty(len(steps_arr))
        energybar_MALA[0] = init_energy
    
        energybar0_MALA = np.empty(len(steps_arr))
        energybar0_MALA[0] = init_energy0
        for i in range(1,len(steps_arr)):
            '''
            beta = 1
            '''
            # proposal = (1-a*stepsize) * X_MALA + np.sqrt(2*stepsize) * np.random.randn(n_samples)
            # acceptance_probs = np.minimum(1, np.exp(-f(proposal) - (X_MALA - proposal + stepsize * df(proposal))**2/(4*stepsize)\
            #                                     + f(X_MALA) + (proposal - X_MALA + stepsize*df(X_MALA))**2 / (4*stepsize)) )
            # acceptance = np.where(np.random.rand(n_samples) <= acceptance_probs, 1, 0)

            # X_MALA = proposal * acceptance + X_MALA * (1-acceptance)

            '''
            experimental b != 1
            '''
            proposal = (1-a*stepsize) * X_MALA + np.sqrt(2*beta*stepsize) * np.random.randn(n_samples)
            acceptance_probs = np.minimum(1, np.exp(-f(proposal)/beta - (X_MALA - proposal + stepsize * df(proposal))**2/(4*stepsize*beta)\
                                                + f(X_MALA)/beta + (proposal - X_MALA + stepsize*df(X_MALA))**2 / (4*stepsize*beta)) )
            acceptance = np.where(np.random.rand(n_samples) <= acceptance_probs, 1, 0)
            X_MALA = proposal * acceptance + X_MALA * (1-acceptance)

            mubar_MALA[i] = np.mean(X_MALA)
            varbar_MALA[i] = np.var(X_MALA) # Biased estimate. 
            energybar_MALA[i] = compute_KL(X_MALA)
            energybar0_MALA[i] = compute_KL(X_MALA, T_target=0)

        print("MALA empirical", time.time()-start)
    #%%
    if WP_ON:
        '''
        Kernel 1: Exact rho_T (T=0 OK)
        '''
        # start = time.time()

        # mubar_KER1 = np.empty((len(WP_T_arr), len(steps_arr)))
        # varbar_KER1 = np.empty((len(WP_T_arr), len(steps_arr)))

        # mubar_KER1[:,0] = np.mean(X_inits)
        # varbar_KER1[:,0] = np.var(X_inits)

        # for ctr, T in enumerate(WP_T_arr):
        #     X_KER1 = X_inits[:,None]
        #     for i in range(1, len(steps_arr)):
        #         def score(x): # grad log_T 
        #             mu_true = mu_WP_arr[ctr, i-1]
        #             var_true = var_WP_arr[ctr, i-1]
        #             return -(x - mu_true)/var_true
        #         X_KER1 = wass_opt_lib.update_once_givenscore(X_KER1,df,beta,stepsize, score)
        #         mubar_KER1[ctr, i] = np.mean(X_KER1)
        #         varbar_KER1[ctr, i] = np.var(X_KER1)

        # print("WP score with exact rho_T", time.time()-start)
        '''
        Kernel 2: Empirical score
        '''
        start = time.time()

        mubar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
        varbar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))

        mubar_KER2[:,0] = np.mean(X_inits)
        varbar_KER2[:,0] = np.var(X_inits)

        energybar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
        energybar_KER2[:,0] = init_energy
            
        energybar0_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
        energybar0_KER2[:,0] = init_energy0
        for ctr, T in enumerate(WP_T_arr):
            X_KER2 = X_inits[:,None]
            for i in range(1, len(steps_arr)):
                X_KER2 = wass_opt_lib.update_once(X_KER2,f,df,beta,T,stepsize)
                mubar_KER2[ctr, i] = np.mean(X_KER2)
                varbar_KER2[ctr, i] = np.var(X_KER2)
                energybar_KER2[ctr,i] = compute_KL(X_KER2,T,T)
                energybar0_KER2[ctr,i] = compute_KL(X_KER2,T,0)
            # if np.var(X_KER2) < 0:
            #     raise Exception("??")

        print("WP kernel score formulation", time.time()-start)
        '''
        Kernel 3: Gaussian approximation rho_T
        '''
        start = time.time()

        mubar_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))
        varbar_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))

        mubar_KER3[:,0] = np.mean(X_inits)
        varbar_KER3[:,0] = np.var(X_inits)
        energybar_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))
        energybar_KER3[:,0] = init_energy
        energybar0_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))
        energybar0_KER3[:,0] = init_energy0
        for ctr, T in enumerate(WP_T_arr):
            X_KER3 = X_inits[:,None]
            for i in range(1, len(steps_arr)):
                mu_app = mubar_KER3[ctr, i-1]
                var_app = varbar_KER3[ctr, i-1]
                # mu_T = (1-a*stepsize + (stepsize*beta*(1+a*WP_T)*a*WP_T)/(var_app + 2*beta*WP_T*(1+a*WP_T)))*mu_app
                # var_T = (1-a*stepsize + (stepsize*beta*(1+a*WP_T)**2)/(var_app + 2*beta*WP_T*(1+a*WP_T)))**2*var_app
                # compute parameters of rho_T
                mu_T = mu_app/(1+a*T)
                var_T = var_app/((1+a*T)**2)+(2*beta*T/(1+a*T))

                
                def score(x): # grad log_T 
                    return -(x - mu_T)/var_T
                X_KER3 = wass_opt_lib.update_once_givenscore(X_KER3,df,beta,stepsize, score)

                mubar_KER3[ctr, i] = np.mean(X_KER3)
                varbar_KER3[ctr, i] = np.var(X_KER3)
                energybar_KER3[ctr,i] = compute_KL(X_KER3,T,T)
                energybar0_KER3[ctr,i] = compute_KL(X_KER3,T,0)
        print("WP gaussian kernel", time.time()-start)
        print("WP mu", mubar_KER3[:,-3:])
        print("WP var", varbar_KER3[:,-3:])
    # '''
    # Kernel 3: Gaussian approximation rho_0
    # '''
    # start = time.time()

    # mubar_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))
    # varbar_KER3 = np.empty((len(WP_T_arr), len(steps_arr)))

    # mubar_KER3[:,0] = np.mean(X_inits)
    # varbar_KER3[:,0] = np.var(X_inits)

    # for ctr, T in enumerate(WP_T_arr):
    #     X_KER3 = X_inits[:,None]
    #     for i in range(1, len(steps_arr)):
    #         def score(x): # grad log_T 
    #             mu_app = mubar_KER3[ctr, i-1]
    #             var_app = varbar_KER3[ctr, i-1]
    #             return -(X_KER3 - mu_app)/var_app
    #         X_KER3 = wass_opt_lib.update_once_givenscore(X_KER3,df,beta,stepsize, score)

    #     mubar_KER3[ctr, i] = np.mean(X_KER3)
    #     varbar_KER3[ctr, i] = np.var(X_KER3)

    # print("WP gaussian kernel rho_T", time.time()-start)

    #%%

    # plotting
    if SGD_ON:
        ax_mu.plot(steps_arr,mubar_sgd, label='SGD empirical')
        ax_var.plot(steps_arr,varbar_sgd, label='SGD empirical')
    if MALA_ON:
        ax_mu.plot(steps_arr,mubar_MALA, label='MALA empirical')
        ax_var.plot(steps_arr,varbar_MALA, label='MALA empirical')

    ax_mu.get_legend().remove()
    ax_var.get_legend().remove()
    ax_mu.legend()
    ax_var.legend()
    fig_mu.savefig(os.path.join(figs_path, "mu_withempirical"))
    fig_var.savefig(os.path.join(figs_path, "var_withempirical"))

    if WP_ON:
        for ctr, T in enumerate(WP_T_arr):
            fig_mu_diffs_WP, ax_mu_diffs_WP = plt.subplots()
            ax_mu_diffs_WP.axhline()
            fig_mu_diffs_WP.suptitle("WProx mu error, T="+str(T))
            # ax_mu_diffs_WP.plot(steps_arr, mubar_KER1[ctr, :] - true_mu, label = "WProx exact")
            ax_mu_diffs_WP.plot(steps_arr, mubar_KER2[ctr, :] - true_mu, label = "WProx kernel")
            ax_mu_diffs_WP.plot(steps_arr, mubar_KER3[ctr, :] - true_mu, label = "WProx Gaussian")
            ax_mu_diffs_WP.legend()
            fig_mu_diffs_WP.savefig(os.path.join(figs_path, "WP_mudiffs_T"+str(T)+".png"))

            fig_var_diffs_WP, ax_var_diffs_WP = plt.subplots()
            ax_var_diffs_WP.axhline()
            fig_var_diffs_WP.suptitle("WProx var error, T="+str(T))
            # ax_var_diffs_WP.plot(steps_arr, varbar_KER1[ctr, :] - true_var, label = "WProx exact")
            ax_var_diffs_WP.plot(steps_arr, varbar_KER2[ctr, :] - true_var, label = "WProx kernel")
            ax_var_diffs_WP.plot(steps_arr, varbar_KER3[ctr, :] - true_var, label = "WProx Gaussian")
            ax_var_diffs_WP.legend()
            fig_var_diffs_WP.savefig(os.path.join(figs_path, "WP_vardiffs_T"+str(T)+".png"))



            fig_mu_WP, ax_mu_WP = plt.subplots()
            fig_mu_WP.suptitle("WProx mu, T="+str(T))
            ax_mu_WP.axhline()
            ax_mu_WP.plot(grid_X, true_mu_base, label = "True")
            ax_mu_WP.plot(steps_arr, mu_WP_arr[ctr, :], label = "WProx analytic")
            # ax_mu_WP.plot(steps_arr, mubar_KER1[ctr, :], label = "WProx exact")
            ax_mu_WP.plot(steps_arr, mubar_KER2[ctr, :], label = "WProx kernel")
            ax_mu_WP.plot(steps_arr, mubar_KER3[ctr, :], label = "WProx Gaussian")
            ax_mu_WP.legend()
            fig_mu_WP.savefig(os.path.join(figs_path, "WP_mu_T"+str(T)+".png"))

            fig_var_WP, ax_var_WP = plt.subplots()
            ax_var_WP.axhline(y=target_var)
            fig_var_WP.suptitle("WProx var, T="+str(T))
            ax_var_WP.plot(grid_X, true_var_base, label = "True")
            ax_var_WP.plot(steps_arr, var_WP_arr[ctr, :], label = "WProx analytic")
            # ax_var_WP.plot(steps_arr, varbar_KER1[ctr, :], label = "WProx exact")
            ax_var_WP.plot(steps_arr, varbar_KER2[ctr, :], label = "WProx kernel")
            ax_var_WP.plot(steps_arr, varbar_KER3[ctr, :], label = "WProx Gaussian")
            ax_var_WP.legend()
            fig_var_WP.savefig(os.path.join(figs_path, "WP_var_T"+str(T)+".png"))
            if SGD_ON:
                ax_mu_diffs_WP.plot(steps_arr, mubar_sgd - true_mu, label = "SGD")
                ax_var_diffs_WP.plot(steps_arr, varbar_sgd - true_var, label = "SGD")
            if MALA_ON:
                ax_mu_diffs_WP.plot(steps_arr, mubar_MALA - true_mu, label = "MALA")
                ax_var_diffs_WP.plot(steps_arr, varbar_MALA - true_var, label = "MALA")
            if SGD_ON or MALA_ON:
                ax_mu_diffs_WP.get_legend().remove()
                ax_var_diffs_WP.get_legend().remove()
                ax_mu_diffs_WP.legend()
                fig_mu_diffs_WP.savefig(os.path.join(figs_path, "WP_mudiffs_extra_T"+str(T)+".png"))
                ax_var_diffs_WP.legend()
                fig_var_diffs_WP.savefig(os.path.join(figs_path, "WP_vardiffs_extra_T"+str(T)+".png"))
            


    # save data


    
    data_path = os.path.join("data", "gaussian_exps", experiment_name)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save empiricals

    np.savez(os.path.join(data_path,"params"), a=a,
                        beta=beta,
                        maximum_T=maximum_T,
                        init_mu=init_mu,
                        init_var=init_var,
                        stepsize=stepsize,
                        WP_T_arr=WP_T_arr)
    # SGD
    if not os.path.exists(os.path.join(data_path,"empirical")):
        os.makedirs(os.path.join(data_path,"empirical"))
    if SGD_ON:
        np.save(os.path.join(data_path,"empirical", "SGD"), X_SGD)
        np.save(os.path.join(data_path,"empirical", "energy_SGD"), energybar_sgd)
        np.save(os.path.join(data_path,"empirical", "energy0_SGD"), energybar0_sgd)
    if MALA_ON:
        np.save(os.path.join(data_path,"empirical", "MALA"), X_MALA)
        np.save(os.path.join(data_path,"empirical", "energy_MALA"), energybar_MALA)
        np.save(os.path.join(data_path,"empirical", "energy0_MALA"), energybar0_MALA)
    if WP_ON:
        # np.save(os.path.join(data_path,"empirical", "WP_KER1"), X_KER1)
        np.save(os.path.join(data_path,"empirical", "WP_KER2"), X_KER2)
        np.save(os.path.join(data_path,"empirical", "WP_KER3"), X_KER3)
        np.save(os.path.join(data_path,"empirical", "energy_KER2"), energybar_KER2)
        np.save(os.path.join(data_path,"empirical", "energy_KER3"), energybar_KER3)
        np.save(os.path.join(data_path,"empirical", "energy0_KER2"), energybar0_KER2)
        np.save(os.path.join(data_path,"empirical", "energy0_KER3"), energybar0_KER3)