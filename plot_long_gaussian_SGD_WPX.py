import numpy as np
import matplotlib.pyplot as plt
import os
import wass_opt_lib
import time
import utils
import create_gif
from tqdm import tqdm
# USER VARIABLES

a = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
beta = 1
maximum_T = 100
# isotropic initalization
init_mu = 0
init_var = 1
problem_dim=10

WP_T_arr = [0.01,0.1]
experiment_name = '21_7_energy_0.1'
stepsize = 0.1

# experiment_name = '19_7_energy_0.1'
# stepsize = 0.1

# experiment_name = '19_7_energy_0.5'
# stepsize = 0.5

SGD_ON = True
EMPIRICAL_ON = True
MALA_ON = True
WP_ON = True
ENERGY_ON = True
# BEGIN

V_name = 'long_gaussian'
figs_path = os.path.join('figs', V_name, "init_mu"+str(init_mu)+"var"+str(init_var),"beta"+str(beta), "stepsize"+str(stepsize))
data_path = os.path.join("data", V_name, experiment_name)

if not os.path.exists(figs_path):
    os.makedirs(figs_path)
if not os.path.exists(os.path.join(figs_path, experiment_name)):
    os.makedirs(os.path.join(figs_path, experiment_name))
if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists(os.path.join(data_path,"empirical")):
    os.makedirs(os.path.join(data_path,"empirical"))


np.savez(os.path.join(data_path,"params"), 
                    beta=beta,
                    maximum_T=maximum_T,
                    init_mu=init_mu,
                    init_var=init_var,
                    stepsize=stepsize,
                    WP_T_arr=WP_T_arr,
                    problem_dim=problem_dim)



def V(x):
    return np.sum(x**2*a[None,:], axis=1)/2

def dV(x):
    return x*a[None,:]
def V_forplotter(x):
    a_forplotter = a[[0,-1]]
    return np.sum(x**2*a_forplotter[None,:], axis=1)/2

def compute_KL(x, T=0.05):
    # computes the KL divergence between samples x and WProx of target distribution
    # missing the normalizing constant for the target distribution.
    if len(x.shape) == 1:
        x = x[:,None]
    if T<= 0:
        raise Exception("T should be positive")


    # gaussian_normalizing_const = np.sqrt(2*np.pi*t_var)
    energy = wass_opt_lib.compute_energy_standalone_corrected(x, V, beta, T, sample_iters=25)
    # energy = energy/len(x) - np.log(len(x)) + np.log(gaussian_normalizing_const)
    return energy
    # compute


# grid_X = np.linspace(0, maximum_T, 201, endpoint=True)

# true_mu_base = init_mu * np.exp(-a*grid_X)
# true_var_base = init_var * np.exp(-2*a*grid_X) + beta*(1-np.exp(-2*a*grid_X))/a
index_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)
steps_arr = np.arange(0, np.floor(maximum_T/stepsize)+1)*stepsize

#%%
'''
Empirical
'''
if EMPIRICAL_ON:
    n_samples = 200

    X_inits = np.random.randn(n_samples, problem_dim) * np.sqrt(init_var) + init_mu
    init_energy = compute_KL(X_inits)
    print("init energy", init_energy)

    '''
    SGD
    '''
    if SGD_ON:
        if not os.path.exists(os.path.join(figs_path, experiment_name, 'SGD')):
            os.makedirs(os.path.join(figs_path, experiment_name, 'SGD'))
        plotter = utils.Plotter2D(os.path.join(figs_path, experiment_name, 'SGD'), V_forplotter, np.linspace(-4,4,51),np.linspace(-4,4,51))
        
        start = time.time()

        energybar_sgd = np.empty(len(steps_arr))
        # mubar_sgd = np.empty(len(steps_arr))
        # varbar_sgd = np.empty(len(steps_arr))
        X_SGD = X_inits
        plotter.do_plotting(X_SGD[:,[0,-1]], name = "0", title="ULA iter 0")         
        # mubar_sgd[0] = np.mean(X_SGD)
        # varbar_sgd[0] = np.var(X_SGD)
        energybar_sgd[0] = init_energy

        for i in tqdm(range(1,len(steps_arr))):
            X_SGD =  X_SGD - stepsize*dV(X_SGD) + np.sqrt(2*beta*stepsize) * np.random.randn(n_samples, problem_dim)
            # mubar_sgd[i] = np.mean(X_SGD)
            # varbar_sgd[i] = np.var(X_SGD) # Biased estimate. 
            energybar_sgd[i] = compute_KL(X_SGD)
            if i%50 == 0:
                plotter.do_plotting(X_SGD[:,[0,-1]], name = str(i), title="ULA iter "+str(i))       
        print("SGD empirical", time.time()-start)
        print(energybar_sgd[-1])

        create_gif.create_gif_from_zoom(os.path.join(figs_path, experiment_name, 'SGD'), name="_SGD", gif_dir=os.path.join(figs_path, experiment_name))
        create_gif.create_gif_from_notzoom(os.path.join(figs_path, experiment_name, 'SGD'), name="_SGD_notzoom", gif_dir=os.path.join(figs_path, experiment_name))
        np.save(os.path.join(data_path,"empirical", "SGD"), X_SGD)
        np.save(os.path.join(data_path,"empirical", "energy_SGD"), energybar_sgd)
    
    #%%
    '''
    MALA
    '''
    if MALA_ON:
        start = time.time()
        if not os.path.exists(os.path.join(figs_path, experiment_name, 'MALA')):
            os.makedirs(os.path.join(figs_path, experiment_name, 'MALA'))
        plotter = utils.Plotter2D(os.path.join(figs_path, experiment_name, 'MALA'), V_forplotter, np.linspace(-4,4,51),np.linspace(-4,4,51))



        if beta != 1:
            print("beta not 1, care")
        mubar_MALA = np.empty(len(steps_arr))
        varbar_MALA = np.empty(len(steps_arr))
        X_MALA = X_inits
        plotter.do_plotting(X_MALA[:,[0,-1]], name = "0", title="MALA iter 0")  
        mubar_MALA[0] = np.mean(X_MALA)
        varbar_MALA[0] = np.var(X_MALA)
        energybar_MALA = np.empty(len(steps_arr))
        energybar_MALA[0] = init_energy

        for i in tqdm(range(1,len(steps_arr))):
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
            proposal = X_MALA - stepsize*dV(X_MALA)+ np.sqrt(2*beta*stepsize) * np.random.randn(n_samples, problem_dim)
            acceptance_probs = np.minimum(1, np.exp(-V(proposal)/beta - np.sum((X_MALA - proposal + stepsize*dV(proposal))**2,axis=1)/(4*stepsize*beta)\
                                                + V(X_MALA)/beta + np.sum((proposal - X_MALA + stepsize*dV(X_MALA))**2,axis=1) / (4*stepsize*beta)))
            acceptance = np.where(np.random.rand(n_samples) <= acceptance_probs, 1, 0)
            X_MALA = proposal * acceptance[:,None] + X_MALA * (1-acceptance[:,None])

            mubar_MALA[i] = np.mean(X_MALA)
            varbar_MALA[i] = np.var(X_MALA) # Biased estimate. 
            energybar_MALA[i] = compute_KL(X_MALA)
            if i%50 == 0:
                plotter.do_plotting(X_MALA[:,[0,-1]], name = str(i), title="MALA iter "+str(i))   

        print("MALA empirical", time.time()-start)
        print(energybar_MALA[-1])
        create_gif.create_gif_from_zoom(os.path.join(figs_path, experiment_name, 'MALA'), name="_MALA", gif_dir=os.path.join(figs_path, experiment_name))
        create_gif.create_gif_from_notzoom(os.path.join(figs_path, experiment_name, 'MALA'), name="_MALA_notzoom", gif_dir=os.path.join(figs_path, experiment_name))
        np.save(os.path.join(data_path,"empirical", "MALA"), X_MALA)
        np.save(os.path.join(data_path,"empirical", "energy_MALA"), energybar_MALA)
    #%%
    if WP_ON:

        '''
        Kernel 2: Empirical score
        '''
        start = time.time()

        

            

        energybar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
        energybar_KER2[:,0] = init_energy
            
        for ctr, T in enumerate(WP_T_arr):
            if not os.path.exists(os.path.join(figs_path, experiment_name, 'WProx'+str(T))):
                os.makedirs(os.path.join(figs_path, experiment_name, 'WProx'+str(T)))

            plotter = utils.Plotter2D(os.path.join(figs_path, experiment_name, 'WProx'+str(T)), V_forplotter, np.linspace(-4,4,51),np.linspace(-4,4,51))
           
            X_KER2 = X_inits
            plotter.do_plotting(X_KER2[:,[0,-1]], name = "0", title="WProx iter 0")  
            for i in tqdm(range(1, len(steps_arr))):
                X_KER2 = wass_opt_lib.update_once(X_KER2,V,dV,beta,T,stepsize)
                energybar_KER2[ctr,i] = compute_KL(X_KER2)
                if i%50 == 0:
                    plotter.do_plotting(X_KER2[:,[0,-1]], name = str(i), title="WProx iter "+str(i))   
            # if np.var(X_KER2) < 0:
            #     raise Exception("??")
            create_gif.create_gif_from_zoom(os.path.join(figs_path, experiment_name, 'WProx'+str(T)), name='_WProx'+str(T), gif_dir=os.path.join(figs_path, experiment_name))
            create_gif.create_gif_from_notzoom(os.path.join(figs_path, experiment_name, 'WProx'+str(T)), name='WProx'+str(T)+"_notzoom", gif_dir=os.path.join(figs_path, experiment_name))
            print("WP", energybar_KER2[ctr,-1])
        print("WP kernel score formulation", time.time()-start)
        np.save(os.path.join(data_path,"empirical", "WP_KER2"), X_KER2)
        np.save(os.path.join(data_path,"empirical", "energy_KER2"), energybar_KER2)

