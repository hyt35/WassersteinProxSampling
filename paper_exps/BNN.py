import numpy as np
import torch
import torch.nn as nn
import argparse
import bnn_utils
import os
import wass_opt_lib_torch
import torch.autograd as autograd
from tqdm import tqdm
from functools import partial

# loggers
import logging

device = 'cuda'
base_checkpoint_path = 'uci/checkpoints'

os.makedirs(base_checkpoint_path, exist_ok = True)

dataset_paths = {
    "boston": 'uci/raw_data/housing/data',
    "combined": 'uci/raw_data/power/data',
    "concrete": 'uci/raw_data/concrete/data',
    "wine": 'uci/raw_data/wine/data',
    "kin8nm": 'uci/raw_data/kin8nm/data'
}

train_params = {
    "boston":   {'n_epochs': 50, 'batch_size': 100, 'stepsize': 1e-4},
    "combined": {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1e-4},
    "concrete": {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1e-4},
    "wine":     {'n_epochs': 20, 'batch_size': 100, 'stepsize': 1e-4},
    "kin8nm":   {'n_epochs': 200, 'batch_size': 100, 'stepsize': 1e-4}
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='concrete', help = "dataset type, boston|combined|concrete|wine|kin8nm")
    parser.add_argument('--nn_init_num', type=int, default = 10, help="Number of particles")
    parser.add_argument('--trial', type=int, default=0, help="trial number to average over")
    parser.add_argument('--T', type=float, default=5e-4)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--stepsize', type=float, default=-1.)
    parser.add_argument('--loggingprefix', type=str, default="gridsearch")
    args = parser.parse_args()


    key = args.data
    if args.stepsize > 0:
        train_params[key]['stepsize'] = args.stepsize

    logging.basicConfig(filename='logs/bnn_'+args.loggingprefix+'_'+key+'.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
    logger = logging.getLogger('bnn')
    logger.setLevel(logging.INFO)
    logger.info("experiment {}({}), stepsize {}, T {}, beta {}".format(key, args.trial, train_params[key]['stepsize'], args.T, args.beta))


    print("key", key)
    train_dl, test_dl, num_features, targ_std, targ_mean, data_1k, targ_1k, n_train = bnn_utils.load_dataset(dataset_paths[key], train_params[key]['batch_size'])

    data_1k = data_1k.to(device)
    targ_1k = targ_1k.to(device)
    tbar = tqdm(train_dl, ncols=80, position=0, leave=True)
    tbar2 = tqdm(range(1,train_params[key]['n_epochs']+1), ncols=120,position=1)
    net_list = bnn_utils.initialize_networks(num_features, device = device, nn_init_num = args.nn_init_num)

    print("initialized model")
    net_params = torch.vstack([bnn_utils.model_to_params(model) for model in net_list])

    print("begin train")
    for epoch in tbar2:
        for batch_idx, (dat_, targ_) in enumerate(tbar):
            dat = dat_.to(device)
            targ = targ_.to(device)
            
            # V = partial(bnn_utils.nllhood_posterior, features = dat, y_true = targ, n_features=num_features) # minimize negative logposterior
            V = partial(bnn_utils.nllhood, features = dat, y_true = targ, n_features=num_features,n_train=n_train) # minimize negative loglikelihood

            def dV(params):
                params.requires_grad_(True)
                return autograd.grad(V(params), params, grad_outputs = torch.ones(args.nn_init_num).to(device))[0]
            
            net_params = wass_opt_lib_torch.update_once(net_params, V, dV, args.beta, args.T, stepsize = train_params[key]['stepsize'], sample_iters=10)
            
            # logger
            # print("V", V(net_params))
            # training
            # llhood = torch.mean(V(net_params))
            
            # diffusion

            # checkpointer
            # tbar.set_description('T ({}|{}) | loss {:.4f}'.format(epoch, batch_idx, llhood.item()))
        dat_train = dat
        targ_train = targ
        if epoch % 5 == 0:
            # validation
            mmse_total = 0
            ll_total = 0
            for dat_, targ_ in test_dl:
                dat = dat_.to(device)
                targ = targ_.to(device)
                mmse, ll, neg_log_var = bnn_utils.mmse_logprior(net_params, dat, targ, num_features, targ_std, targ_mean, data_1k, targ_1k)

                mmse_total += mmse.item()
                ll_total += ll.item()
            mmse_full = mmse_total / len(test_dl)
            tbar2.set_description('epoch {}, mmse {:.4f}, rmse {:.4f}, ll {:.4f}'.format(epoch, mmse_full, np.sqrt(mmse_full), ll.item()))
            if np.isnan(mmse_total) or np.isnan(ll_total) or mmse_total > 1000:
                logger.info("NaN")
                raise Exception
            # print("epoch", epoch, "mmse", mmse_full, "rmse", np.sqrt(mmse_full), "llhood", ll.item(), "nlv", neg_log_var.item())

            # checkpoint
            if epoch != 0 and epoch % 20 == 0:
                current_checkpoint_path = os.path.join(base_checkpoint_path, key, str(args.trial), str(epoch))
                os.makedirs(current_checkpoint_path, exist_ok=True)
                torch.save(net_params, os.path.join(current_checkpoint_path,"ckpt.pt"))
                # print("checkpointed epoch", str(epoch))

        if epoch % (train_params[key]['n_epochs']/2) == 0 and epoch != 0:
            train_params[key]['stepsize'] = train_params[key]['stepsize'] * 0.1
        # if epoch % (train_params[key]['n_epochs']//3) == 0 and epoch != 0:
        #     train_params[key]['stepsize'] = train_params[key]['stepsize'] * 0.5
    
        # if epoch % (train_params[key]['n_epochs']/5) == 0 and epoch != 0:
        #     train_params[key]['stepsize'] = train_params[key]['stepsize'] * 0.8
        # if epoch % (train_params[key]['n_epochs']/5) == 0 and epoch != 0:
        #     train_params[key]['stepsize'] = train_params[key]['stepsize'] * 0.6
        tbar.reset()
    logger.info("MMSE {}, LL {}".format(np.sqrt(mmse_full), ll.item()))

if __name__ == "__main__":
    main()