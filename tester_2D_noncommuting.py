import numpy as np
import wass_opt_lib
import utils
import os
from tqdm import tqdm 
from argparse import ArgumentParser
import create_gif
parser = ArgumentParser()
parser = utils.add_specific_args(parser)
parser.add_argument('--test_fn', type=str, default="skew_quadratic")
hparams = parser.parse_args()


num_inits = 400
TEST_FN = hparams.test_fn
# TEST_FN = "wavy" 
beta = 1
T = 0.5
stepsize = 0.1


n_iters = hparams.n_iters
sample_iters = hparams.sample_iters # for approximating the normalizing constant

Sigma = np.asarray([[2,-1],[-1,2]]) 
Sigma_inv = np.linalg.inv(Sigma)
def skew_quadratic(x):
    return np.einsum("ij,jk,ik->i",x,Sigma_inv,x)/2
def grad_skew_quadratic(x):
    return np.einsum("jk,ik->ij",Sigma_inv,x)
def exp_skew_quadratic(x):
    return np.exp(-skew_quadratic(x))
if TEST_FN == "skew_quadratic":
    V = skew_quadratic
    dV = grad_skew_quadratic
    figs_path_base = os.path.join("figs", "tester2D", TEST_FN)


# plotter = utils.Plotter2D(figs_path, V, np.linspace(-4,4,51),np.linspace(-4,4,51))
# plotter.do_plotting(0, name="test")


# initializer
def initer():
    foo = np.random.randn(num_inits, 2)
    foo[:,0] = foo[:,0] * 2
    return foo
X_list = initer() # gaussian
# X_list = (np.random.rand(num_inits, 2)-0.5) * 4 # uniform [-3,3]


#%%
'''
Wass opt
dV (1)
'''
print("dV")



figs_path = os.path.join(figs_path_base, "dV", "beta" + str(beta) + "_T"+str(T)+"_ss"+str(stepsize))
if TEST_FN == "skew_quadratic" or TEST_FN == "minus_double_banana":
    plotter = utils.Plotter2D(figs_path, exp_skew_quadratic, np.linspace(-4,4,51),np.linspace(-4,4,51))
else: 
    raise Exception("plotter undefined testfn")
plotter.do_plotting(X_list, name = "init")


for ctr in tqdm(range(n_iters)):
    X_list = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters)
    # print(ctr, X_list.shape)
    plotter.do_plotting(X_list, name = str(ctr), title="WProx iter "+str(ctr))
    
# for ctr in tqdm(range(5000)):
#     X_list = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters)
#     # print(ctr, X_list.shape)
#     if ctr % 100 == 99:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title="backward iter " + str(ctr+n_iters+1))
create_gif.create_gif_from_zoom(figs_path, name="_WProx", gif_dir=figs_path)
print(np.cov(X_list.T))
# #%%
# '''
# Wass opt
# dPhi (2)
# dX/dT = dPhi(t,X) - beta dlog rho(t,X)
# discretized as dPhi(0,X), dlog rho(T,X)
# '''
# print("dPhi")
# X_list = initer() # gaussian
# beta = hparams.beta
# T = hparams.T
# stepsize = hparams.stepsize


# figs_path = os.path.join(figs_path_base, "dPhi0", "beta" + str(beta) + "_T"+str(T)+"_ss"+str(stepsize))
# if TEST_FN == "skew_quadratic" or TEST_FN == "minus_double_banana":
#     plotter = utils.Plotter2D(figs_path, exp_skew_quadratic, np.linspace(-4,4,51),np.linspace(-4,4,51))
# else: 
#     raise Exception("plotter undefined testfn")
# plotter.do_plotting(X_list, name = "init")


# for ctr in tqdm(range(n_iters)):
#     X_list = wass_opt_lib.update_once_Phi(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters)
#     # print(ctr, X_list.shape)
#     plotter.do_plotting(X_list, name = str(ctr), title="forward iter " + str(ctr))
    
# for ctr in tqdm(range(5000)):
#     X_list = wass_opt_lib.update_once_Phi(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters)
#     # print(ctr, X_list.shape)
#     if ctr % 100 == 99:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title = "forward iter"+ str(ctr+n_iters+1))
#%%
'''
SGD tester
'''

X_list = initer() # gaussian
figs_path = os.path.join("figs", "tester2D", TEST_FN, "SGD_beta"+str(beta)+"_ss"+str(stepsize))
if TEST_FN == "skew_quadratic" or TEST_FN == "minus_double_banana":
    plotter = utils.Plotter2D(figs_path, exp_skew_quadratic, np.linspace(-4,4,51),np.linspace(-4,4,51))
else: 
    raise Exception("plotter undefined testfn")
plotter.do_plotting(X_list, name = "init")


for ctr in tqdm(range(n_iters)):
    X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
    # print(ctr, X_list.shape)
    plotter.do_plotting(X_list, name = str(ctr), title="ULA iter "+str(ctr))
    
# for ctr in tqdm(range(5000)):
#     X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
#     # print(ctr, X_list.shape)
#     if ctr % 100 == 99:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title="sgd iter " + str(ctr+n_iters+1))
print(np.cov(X_list.T))
create_gif.create_gif_from_zoom(figs_path, name="_SGD", gif_dir=figs_path)