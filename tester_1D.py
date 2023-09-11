import numpy as np
import wass_opt_lib
import utils
import os
from tqdm import tqdm 
from argparse import ArgumentParser

parser = ArgumentParser()
parser = utils.add_specific_args(parser)
parser.add_argument('--test_fn', type=str, default="exp_wavy")
hparams = parser.parse_args()


num_inits = hparams.num_inits
TEST_FN = hparams.test_fn
# TEST_FN = "wavy" 
wavy_T = 2


n_iters = hparams.n_iters
sample_iters = hparams.sample_iters # for approximating the normalizing constant


def wavy(x, T=wavy_T):
    x = x.ravel()
    return (x**2 - T*np.cos(2*np.pi*x)+T)

def grad_wavy(x, T=wavy_T):
    return (2*x + 2*np.pi*T*np.sin(2*np.pi*x))


def twopole(x):
    x = x.ravel()
    return x**4 - x**2 - x/4

def grad_twopole(x):
    return 4*x**3 - 2*x - 1/4

def exp_twopole(x):
    return -np.exp(-twopole(x))

def grad_exp_twopole(x):
    return grad_twopole(x) * np.exp(-twopole(x))[:,None]

def exp_wavy(x, T=wavy_T):
    # foo = -np.exp(-wavy(x, T))
    # print("e", foo.shape)
    return -np.exp(-wavy(x, T)).ravel()

def grad_exp_wavy(x, T=wavy_T):
    # foo = grad_wavy(x, T) * np.exp(-wavy(x, T))
    # bar = grad_wavy(x, T)
    # print("ge", x.shape, foo.shape, bar.shape)
    return grad_wavy(x, T) * np.exp(-wavy(x, T))[:,None] # extra dimension to fix broadcasting problems

if TEST_FN == "wavy":
    V = wavy
    dV = grad_wavy
    figs_path_base = os.path.join("figs", "tester", TEST_FN+str(wavy_T))
elif TEST_FN == "exp_wavy":
    V = exp_wavy
    dV = grad_exp_wavy
    figs_path_base = os.path.join("figs", "tester", TEST_FN+str(wavy_T))
elif TEST_FN == "twopole":
    V = twopole
    dV = grad_twopole
    figs_path_base = os.path.join("figs", "tester", TEST_FN)
elif TEST_FN == "exp_twopole":
    V = exp_twopole
    dV = grad_exp_twopole
    figs_path_base = os.path.join("figs", "tester", TEST_FN)




# initializer
def initer():
    return np.random.randn(num_inits, 1)

# X_list = (np.random.rand(num_inits, 1)-0.5) * 4 # uniform [-3,3]


#%%
# X_list = initer()
# beta = hparams.beta
# T = hparams.T
# stepsize = hparams.stepsize


# figs_path = os.path.join(figs_path_base,"dV",  "beta" + str(beta) + "_T"+str(T)+"_ss"+str(stepsize))
# if TEST_FN == "wavy" or TEST_FN == "exp_wavy":
#     plotter = utils.Plotter1D(figs_path, V, np.linspace(-5,5,201))
# elif TEST_FN == "twopole" or TEST_FN == "exp_twopole":
#     plotter = utils.Plotter1D(figs_path, V, np.linspace(-1,1,201))
# else: 
#     raise Exception("plotter undefined testfn")
# plotter.do_plotting(X_list, name = "init")


# for ctr in tqdm(range(n_iters)):
#     X_list = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize)
#     # print(ctr, X_list.shape)
#     plotter.do_plotting(X_list, name = str(ctr))
    
# for ctr in tqdm(range(5000)):
#     X_list = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize)
#     # print(ctr, X_list.shape)
#     if ctr % 100 == 99:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1))

# #%%
# X_list = initer()
# beta = hparams.beta
# T = hparams.T
# stepsize = hparams.stepsize


# figs_path = os.path.join(figs_path_base, "dPhi0", "beta" + str(beta) + "_T"+str(T)+"_ss"+str(stepsize))
# if TEST_FN == "wavy" or TEST_FN == "exp_wavy":
#     plotter = utils.Plotter1D(figs_path, V, np.linspace(-5,5,201))
# elif TEST_FN == "twopole" or TEST_FN == "exp_twopole":
#     plotter = utils.Plotter1D(figs_path, V, np.linspace(-1,1,201))
# else: 
#     raise Exception("plotter undefined testfn")
# plotter.do_plotting(X_list, name = "init")


# for ctr in tqdm(range(n_iters)):
#     X_list = wass_opt_lib.update_once_Phi(X_list, V, dV, beta, T, stepsize, sample_iters = sample_iters)
#     # print(ctr, X_list.shape)
#     plotter.do_plotting(X_list, name = str(ctr))
    
# for ctr in tqdm(range(5000)):
#     X_list = wass_opt_lib.update_once_Phi(X_list, V, dV, beta, T, stepsize, sample_iters = sample_iters)
#     # print(ctr, X_list.shape)
#     if ctr % 100 == 99:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1))



#%%
'''
SGD tester
'''
beta = hparams.beta
T = hparams.T
stepsize = hparams.stepsize

X_list = initer() # gaussian
figs_path = os.path.join(figs_path_base, "SGD_beta"+str(beta)+"_ss"+str(stepsize))
if TEST_FN == "wavy" or TEST_FN == "exp_wavy":
    plotter = utils.Plotter1D(figs_path, V, np.linspace(-5,5,201))
elif TEST_FN == "twopole" or TEST_FN == "exp_twopole":
    plotter = utils.Plotter1D(figs_path, V, np.linspace(-1,1,201))
else: 
    raise Exception("plotter undefined testfn")
plotter.do_plotting(X_list, name = "init")


for ctr in tqdm(range(n_iters)):
    X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
    # print(ctr, X_list.shape)
    plotter.do_plotting(X_list, name = str(ctr), title="sgd iter " + str(ctr))
    
for ctr in tqdm(range(5000)):
    X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
    # print(ctr, X_list.shape)
    if ctr % 100 == 99:
        plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title="sgd iter " + str(ctr+n_iters+1))
