import numpy as np
import wass_opt_lib
import utils
import os
from tqdm import tqdm 
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import create_gif
import time

# Plotting script: bayesian_logreg_plotter.py
# python bayesian_logreg.py --num_inits=100 --beta=1 --stepsize=0.1 --sample_iters=25 --n_iters=5000 --problem_dim=2 --logistic_samples=50 --alpha=0.5
# python bayesian_logreg.py --num_inits=100 --beta=1 --stepsize=0.05 --sample_iters=25 --n_iters=5000 --problem_dim=2 --logistic_samples=50 --alpha=0.5
# T is hardcoded to be [0.05,0.1,0.2]
def generate_pairs(path, problem_dim, num_samples):
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.isfile(os.path.join(path, str(problem_dim)+'_'+str(num_samples)+".npz")):
        # load
        foo = np.load(os.path.join(path, str(problem_dim)+'_'+str(num_samples)+".npz"))
        print("Found file, length", str(len(foo['y'])))
        return foo # should be a dictionary of x and y
    else:
        if num_samples is None:
            raise Exception("File not found, need to pass number of samples")
        print("Creating for num_dims", str(problem_dim), "num_samples", str(num_samples))
        x = 2*np.random.binomial(1, 0.5, size=(num_samples, problem_dim))-1 # Rademacher
        theta_star = np.ones(problem_dim)
        # Sample response
        theta_T_x = np.dot(x, theta_star)
        # random
        y = np.random.binomial(1, np.exp(theta_T_x)/(1+np.exp(theta_T_x)))
        cov = np.inner(x.T,x.T)/num_samples
        np.savez(os.path.join(path, str(problem_dim)+'_'+str(num_samples)), x=x, y=y, cov=cov)
        foo = {"x":x, "y":y, "cov":cov}
        print("Created")
        return foo

parser = ArgumentParser()
parser = utils.add_specific_args(parser)
parser.add_argument('--test_fn', type=str, default="logreg")
parser.add_argument('--problem_dim', type=int, default=2)
parser.add_argument('--logistic_samples', type=int, default=50)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--regenerate', type=bool, default=False)
hparams = parser.parse_args()

num_inits = hparams.num_inits
TEST_FN = hparams.test_fn
n_iters = hparams.n_iters
sample_iters = hparams.sample_iters # for approximating the normalizing constant
beta = hparams.beta
stepsize = hparams.stepsize
problem_dim = hparams.problem_dim
logistic_samples = hparams.logistic_samples
alpha = hparams.alpha
logi_data_path = os.path.join("data", "logreg")
data_path = os.path.join("data", "logreg", str(problem_dim)+'_'+str(logistic_samples)+'_'+str(alpha),"beta"+str(beta)+"ss"+str(stepsize))
WP_T_arr = [0.025,0.05,0.1,0.2]
SGD_ON = True
MALA_ON = True
WP_ON = True
# generate
if hparams.regenerate and os.path.exists(os.path.join(logi_data_path, str(problem_dim)+'_'+str(logistic_samples)+".npz")):
    print("Regenerating data")
    os.remove(os.path.join())
if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists(os.path.join(data_path,"empirical")):
    os.makedirs(os.path.join(data_path,"empirical"))
# contains x and y and cov, x.shape = [n,d], y.shape = [d]
# cov.shape = [d,d]
foo = generate_pairs(logi_data_path, problem_dim, logistic_samples)
if len(foo['y']) != logistic_samples:
    print("Length of loaded array is ", str(len(foo['y'])), "different from parsed", str(logistic_samples))
    logistic_samples = len(foo['y'])


def logreg(theta):
    # theta.shape = [N,d]
    # x.shape = [n,d]
    # y.shape = [d]
    # f(theta) = -Y^T X theta + sum_i log(1+exp(theta^T x_i)) + alpha ||Sigma_X^1/2 theta||_2^2
    theta_T_x = np.inner(foo['x'], theta) # [n, N]
    term_1 = np.dot(foo['y'], theta_T_x) # [N]
    term_2 = np.sum(np.log(1+np.exp(theta_T_x)), axis=0) # [N]
    term_3 = alpha * np.einsum('ij,jk,ik->i',theta,foo['cov'],theta) # [N]
    return -term_1 + term_2 + term_3 # [N]

def grad_logreg(theta):
    x = foo['x']
    theta_T_x = np.inner(foo['x'], theta) # [n, N]
    term_1 = np.dot(foo['x'].T, foo['y']) # [d]
    term_2 = np.sum(x[:,:,None]/np.expand_dims(1+np.exp(-theta_T_x), axis=1), axis=0) # [d,N]
    term_3 = alpha * (foo['cov'].T @ theta.T).T # [N,d]
    return -term_1 + term_2.T + term_3 # [N,d]

def theta_err(theta):
    return np.mean(np.abs(theta-1))

def theta_err2(theta):
    return np.mean(np.abs(np.mean(theta, axis=0)-1))

if TEST_FN == "logreg":
    V = logreg
    dV = grad_logreg
    figs_path_base = os.path.join("figs", TEST_FN, str(problem_dim)+'_'+str(logistic_samples)+'_'+str(alpha),"beta"+str(beta)+"ss"+str(stepsize))

cov_eigs = np.linalg.eigvals(foo['cov'])
L = (0.25 * logistic_samples + alpha) * np.max(cov_eigs)
m = alpha * np.min(cov_eigs)

print("lambda_max", str(L), "lambda_min", str(m))
print("condition number", str(L/m))

np.savez(os.path.join(data_path,"params"), 
                    beta=beta,
                    n_iters=n_iters,
                    stepsize=stepsize,
                    WP_T_arr=WP_T_arr)

# initializer
# TODO fix
def initer():
    return np.random.randn(num_inits, problem_dim)/np.sqrt(L)
X_list = initer() # gaussian


X_inits = X_list
steps_arr = np.arange(0, n_iters+1)
init_err = theta_err(X_inits)
init_err2 = theta_err2(X_inits)
'''
SGD
'''
if SGD_ON:
    if not os.path.exists(os.path.join(figs_path_base, 'SGD')):
        os.makedirs(os.path.join(figs_path_base, 'SGD'))
    if problem_dim == 2:
        plotter = utils.Plotter2D(os.path.join(figs_path_base, 'SGD'), None, np.linspace(0,2,21),np.linspace(0,2,21))
    
    start = time.time()

    errbar_sgd = np.empty(len(steps_arr))
    err2bar_sgd = np.empty(len(steps_arr))
    # mubar_sgd = np.empty(len(steps_arr))
    # varbar_sgd = np.empty(len(steps_arr))
    X_SGD = X_inits
    if problem_dim == 2:
        plotter.do_plotting(X_SGD, name = "0", title="ULA iter 0")         
    # mubar_sgd[0] = np.mean(X_SGD)
    # varbar_sgd[0] = np.var(X_SGD)
    errbar_sgd[0] = init_err
    err2bar_sgd[0] = init_err2
    for i in tqdm(range(1,len(steps_arr))):
        X_SGD =  X_SGD - stepsize*dV(X_SGD) + np.sqrt(2*beta*stepsize) * np.random.randn(num_inits, problem_dim)
        # mubar_sgd[i] = np.mean(X_SGD)
        # varbar_sgd[i] = np.var(X_SGD) # Biased estimate. 
        errbar_sgd[i] = theta_err(X_SGD)
        err2bar_sgd[i] = theta_err2(X_SGD)
        if i%50 == 0 and problem_dim == 2:
            plotter.do_plotting(X_SGD, name = str(i), title="ULA iter "+str(i))       
    print("SGD empirical", time.time()-start)
    print(errbar_sgd[-1])
    print(err2bar_sgd[-1])
    if problem_dim == 2:
        create_gif.create_gif_from_zoom(os.path.join(figs_path_base,  'SGD'), name="_SGD", gif_dir=os.path.join(figs_path_base))
        create_gif.create_gif_from_notzoom(os.path.join(figs_path_base,  'SGD'), name="_SGD_notzoom", gif_dir=os.path.join(figs_path_base))
    np.save(os.path.join(data_path,"empirical", "SGD"), X_SGD)
    np.save(os.path.join(data_path,"empirical", "err_SGD"), errbar_sgd)
    np.save(os.path.join(data_path,"empirical", "err2_SGD"), err2bar_sgd)
#%%
'''
MALA
'''
if MALA_ON:
    start = time.time()
    if not os.path.exists(os.path.join(figs_path_base,  'MALA')):
        os.makedirs(os.path.join(figs_path_base,  'MALA'))
    if problem_dim == 2:
        plotter = utils.Plotter2D(os.path.join(figs_path_base, 'MALA'), None, np.linspace(0,2,21),np.linspace(0,2,21))


    if beta != 1:
        print("beta not 1, care")
    # mubar_MALA = np.empty(len(steps_arr))
    # varbar_MALA = np.empty(len(steps_arr))
    X_MALA = X_inits
    if problem_dim == 2:
        plotter.do_plotting(X_MALA, name = "0", title="MALA iter 0")  
    # mubar_MALA[0] = np.mean(X_MALA)
    # varbar_MALA[0] = np.var(X_MALA)
    errbar_MALA = np.empty(len(steps_arr))
    errbar_MALA[0] = init_err
    err2bar_MALA = np.empty(len(steps_arr))
    err2bar_MALA[0] = init_err2
    for i in tqdm(range(1,len(steps_arr))):
        proposal = X_MALA - stepsize*dV(X_MALA)+ np.sqrt(2*beta*stepsize) * np.random.randn(num_inits, problem_dim)
        # print(proposal.shape)
        # print(X_MALA.shape)
        acceptance_probs = np.minimum(1, np.exp(-V(proposal)/beta - np.sum((X_MALA - proposal + stepsize*dV(proposal))**2,axis=1)/(4*stepsize*beta)\
                                            + V(X_MALA)/beta + np.sum((proposal - X_MALA + stepsize*dV(X_MALA))**2,axis=1) / (4*stepsize*beta)))
        acceptance = np.where(np.random.rand(num_inits) <= acceptance_probs, 1, 0)
        X_MALA = proposal * acceptance[:,None] + X_MALA * (1-acceptance[:,None])

        # mubar_MALA[i] = np.mean(X_MALA)
        # varbar_MALA[i] = np.var(X_MALA) # Biased estimate. 
        errbar_MALA[i] = theta_err(X_MALA)
        err2bar_MALA[i] = theta_err2(X_MALA)
        if i%50 == 0 and problem_dim == 2:
            plotter.do_plotting(X_MALA, name = str(i), title="MALA iter "+str(i))   

    print("MALA empirical", time.time()-start)
    print(errbar_MALA[-1])
    print(err2bar_MALA[-1])
    if problem_dim == 2:
        create_gif.create_gif_from_zoom(os.path.join(figs_path_base,  'MALA'), name="_MALA", gif_dir=os.path.join(figs_path_base))
        create_gif.create_gif_from_notzoom(os.path.join(figs_path_base,  'MALA'), name="_MALA_notzoom", gif_dir=os.path.join(figs_path_base))
    np.save(os.path.join(data_path,"empirical", "MALA"), X_MALA)
    np.save(os.path.join(data_path,"empirical", "err_MALA"), errbar_MALA)
    np.save(os.path.join(data_path,"empirical", "err2_MALA"), err2bar_MALA)
#%%
if WP_ON:

    '''
    Kernel 2: Empirical score
    '''
    start = time.time()
    errbar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
    errbar_KER2[:,0] = init_err
    err2bar_KER2 = np.empty((len(WP_T_arr), len(steps_arr)))
    err2bar_KER2[:,0] = init_err2
    for ctr, T in enumerate(WP_T_arr):
        if not os.path.exists(os.path.join(figs_path_base,  'WProx'+str(T))):
            os.makedirs(os.path.join(figs_path_base,  'WProx'+str(T)))

        if problem_dim == 2:
            plotter = utils.Plotter2D(os.path.join(figs_path_base, 'WProx'+str(T)), None, np.linspace(0,2,21),np.linspace(0,2,21))
        X_KER2 = X_inits
        if problem_dim == 2:
            plotter.do_plotting(X_KER2, name = "0", title="WProx iter 0")  
        for i in tqdm(range(1, len(steps_arr))):
            X_KER2 = wass_opt_lib.update_once(X_KER2,V,dV,beta,T,stepsize)
            errbar_KER2[ctr,i] = theta_err(X_KER2)
            err2bar_KER2[ctr,i] = theta_err2(X_KER2)
            if i%50 == 0 and problem_dim == 2:
                plotter.do_plotting(X_KER2, name = str(i), title="WProx iter "+str(i))   
        # if np.var(X_KER2) < 0:
        #     raise Exception("??")
        if problem_dim == 2:
            create_gif.create_gif_from_zoom(os.path.join(figs_path_base,  'WProx'+str(T)), name='_WProx'+str(T), gif_dir=os.path.join(figs_path_base))
            create_gif.create_gif_from_notzoom(os.path.join(figs_path_base,  'WProx'+str(T)), name='WProx'+str(T)+"_notzoom", gif_dir=os.path.join(figs_path_base))
        print("WP", errbar_KER2[ctr,-1])
        print("WP", err2bar_KER2[ctr,-1])
    print("WP kernel score formulation", time.time()-start)
    np.save(os.path.join(data_path,"empirical", "WP_KER2"), X_KER2)
    np.save(os.path.join(data_path,"empirical", "err_KER2"), errbar_KER2)
    np.save(os.path.join(data_path,"empirical", "err2_KER2"), err2bar_KER2)
