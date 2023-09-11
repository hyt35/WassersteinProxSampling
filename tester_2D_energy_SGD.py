import numpy as np
import wass_opt_lib
import utils
import os
from tqdm import tqdm 
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import create_gif

parser = ArgumentParser()
parser = utils.add_specific_args(parser)
parser.add_argument('--test_fn', type=str, default="minus_double_banana")
hparams = parser.parse_args()


num_inits = hparams.num_inits
TEST_FN = hparams.test_fn
# TEST_FN = "wavy" 
wavy_T = 2


n_iters = hparams.n_iters
sample_iters = hparams.sample_iters # for approximating the normalizing constant


def double_banana(x):
    return np.exp(-2*(np.linalg.norm(x, axis=1)-3)**2) * \
        (np.exp(-2*(x[:,0]-3)**2) + np.exp(-2*(x[:,0]+3)**2))

def grad_double_banana(x):
    dx1_term1 = (-4*x[:,0] + 12*x[:,0]/np.linalg.norm(x, axis=1))* double_banana(x)
    dx1_term2 = np.exp(-2*(np.linalg.norm(x, axis=1)-3)**2) * \
            (-4*(x[:,0]-3) * np.exp(-2*(x[:,0]-3)**2) + \
                -4*(x[:,0]+3)*np.exp(-2*(x[:,0]+3)**2) )
    dx2 = (-4*x[:,1] + 12*x[:,1]/np.linalg.norm(x, axis=1))* double_banana(x)
    return np.array([dx1_term1 + dx1_term2, dx2]).T

def minus_double_banana(x):
    return -double_banana(x)

def grad_minus_double_banana(x):
    return -grad_double_banana(x)

if TEST_FN == "double_banana":
    V = double_banana
    dV = grad_double_banana
    figs_path_base = os.path.join("figs", "tester2D_energy", TEST_FN)
elif TEST_FN == "minus_double_banana":
    V = minus_double_banana
    dV = grad_minus_double_banana
    figs_path_base = os.path.join("figs", "tester2D_energy", TEST_FN)

# plotter = utils.Plotter2D(figs_path, V, np.linspace(-4,4,51),np.linspace(-4,4,51))
# plotter.do_plotting(0, name="test")


# initializer
def initer():
    return np.random.randn(num_inits, 2)*2
X_list = initer() # gaussian
# X_list = (np.random.rand(num_inits, 2)-0.5) * 4 # uniform [-3,3]

#TODO add energy
#%%
'''
Wass opt
dV (1)
'''
# print("dV")
# beta = hparams.beta
# T = hparams.T
# stepsize = hparams.stepsize


# figs_path = os.path.join(figs_path_base, "dV", "beta" + str(beta) + "_T"+str(T)+"_ss"+str(stepsize),'dat')
# if TEST_FN == "double_banana" or TEST_FN == "minus_double_banana":
#     plotter = utils.Plotter2D(figs_path, V, np.linspace(-4,4,51),np.linspace(-4,4,51))
# else: 
#     raise Exception("plotter undefined testfn")
# plotter.do_plotting(X_list, name = "init")

# energy_arr = np.zeros(n_iters+50000)

# for ctr in tqdm(range(n_iters)):
#     X_list, energy = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters, compute_energy=True)
#     # print(ctr, X_list.shape)
#     # plotter.do_plotting(X_list, name = str(ctr), title="backward iter " + str(ctr))
#     if ctr % 10 == 9:
#         plotter.do_plotting(X_list, name = str(ctr+1), title="backward iter " + str(ctr+1))
#     energy_arr[ctr] = energy
# # plot energy
# fig, ax = plt.subplots()
# fig.suptitle("energy")
# ax.plot(energy_arr[:100])
# fig.savefig(os.path.join(figs_path, "_energy_short"))
# fig.clf()
# for ctr in tqdm(range(50000)):
#     X_list, energy = wass_opt_lib.update_once(X_list, V, dV, beta, T, stepsize, sample_iters=sample_iters, compute_energy=True)
#     # print(ctr, X_list.shape)
#     if ctr % 500 == 499:
#         plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title="backward iter " + str(ctr+n_iters+1))
#     energy_arr[ctr+n_iters] = energy


# create_gif.create_gif_from_zoom(figs_path, name="_comp", gif_dir=os.path.dirname(figs_path))
# create_gif.create_gif_from_notzoom(figs_path, name="_comp_notzoom", gif_dir=os.path.dirname(figs_path))
# fig2, ax2 = plt.subplots()
# fig2.suptitle("energy")
# ax2.plot(energy_arr)
# fig2.savefig(os.path.join(figs_path, "_energy"))

#%%
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
# if TEST_FN == "double_banana" or TEST_FN == "minus_double_banana":
#     plotter = utils.Plotter2D(figs_path, V, np.linspace(-4,4,51),np.linspace(-4,4,51))
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
# #%%
# '''
# SGD tester
# '''
beta = hparams.beta
T = hparams.T
stepsize = hparams.stepsize

energy_list = np.zeros(n_iters+50000) # why did i change the name here?

X_list = initer() # gaussian
figs_path = os.path.join("figs", "tester2D_energy", TEST_FN, "SGD", "beta"+str(beta)+"_ss"+str(stepsize),'dat')
if TEST_FN == "double_banana" or TEST_FN == "minus_double_banana":
    plotter = utils.Plotter2D(figs_path, V, np.linspace(-4,4,51),np.linspace(-4,4,51))
    
else: 
    raise Exception("plotter undefined testfn")
plotter.do_plotting(X_list, name = "init")


for ctr in tqdm(range(n_iters)):
    X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
    # print(ctr, X_list.shape)
    plotter.do_plotting(X_list, name = str(ctr), title="sgd iter " + str(ctr))
    energy = wass_opt_lib.compute_energy_standalone(X_list, V, beta, T, sample_iters)
    energy_list[ctr] = energy
    if ctr % 10 == 9:
        plotter.do_plotting(X_list, name = str(ctr+1), title="backward iter " + str(ctr+1))

fig3, ax3 = plt.subplots()
fig3.suptitle("energy")
ax3.plot(energy_list[:n_iters])
fig3.savefig(os.path.join(figs_path, "_energy_short"))

for ctr in tqdm(range(50000)):
    X_list = wass_opt_lib.update_once_SGD(X_list, dV, beta, stepsize)
    energy = wass_opt_lib.compute_energy_standalone(X_list, V, beta, T, sample_iters)
    # print(ctr, X_list.shape)
    if ctr % 500 == 499:
        plotter.do_plotting(X_list, name = str(ctr+n_iters+1), title="sgd iter " + str(ctr+n_iters+1))
    energy_list[ctr+n_iters]=energy

create_gif.create_gif_from_zoom(figs_path, name="_comp", gif_dir=os.path.dirname(figs_path))
create_gif.create_gif_from_notzoom(figs_path, name="_comp_notzoom", gif_dir=os.path.dirname(figs_path))

fig4, ax4 = plt.subplots()
fig4.suptitle("energy")
ax4.plot(energy_list)
fig4.savefig(os.path.join(figs_path, "_energy"))