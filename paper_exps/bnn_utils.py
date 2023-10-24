import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
# class UCIDataset(torch.utils.data.Dataset):
#     def __init__(self, df, ind_features, ind_target):


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return 

def load_dataset(data_path, batch_size, train_percentage = 0.9):
    # takes a path containing raw data and returns train and test dataloaders

    # load to dataset
    df = pd.read_table(os.path.join(data_path, "data.txt"), header = None, sep="\s+") # variable length whitespace delimiter

    index_features = np.loadtxt(os.path.join(data_path, "index_features.txt"), dtype=np.int)
    index_target = np.loadtxt(os.path.join(data_path, "index_target.txt"), dtype=np.int)

    df_as_tensor = torch.Tensor(df.values)
    data = df_as_tensor[:,index_features]
    targ = df_as_tensor[:,index_target]

    data_std, data_mean = torch.std_mean(data, dim=0, unbiased=False)
    targ_std, targ_mean = torch.std_mean(targ, dim=0, unbiased=False)
    data_std[data_std == 0] = 1

    data = (data - data_mean[None,:])/data_std[None,:]
    targ = (targ - targ_mean)/targ_std

    data_1k = data[:1000]
    targ_1k = targ[:1000]
    # data = torch.Tensor(df[index_features].values)
    # targ = torch.Tensor(df[index_target].values)

    uci_dataset = torch.utils.data.TensorDataset(data, targ)

    train_count = int(len(df) * train_percentage)
    train_set, test_set = torch.utils.data.random_split(uci_dataset, [train_count, len(df) - train_count])

    # split into train and test dataset
    # print(df)
    # print(index_features)
    # print(index_target)

    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True, drop_last = True)
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=len(df) - train_count, shuffle = True, drop_last = True)

    num_features = len(index_features)

    return train_dl, test_dl, num_features, targ_std, targ_mean, data_1k, targ_1k, train_count

class Net(nn.Module):
    # Basic 1 layer neural network
    # should be sufficient for this.
    def __init__(self, n_features = [5,50,50]):
        super(Net, self).__init__()
        features_plus1 = n_features+[1]
        self.filters = nn.ModuleList([nn.Linear(features_plus1[i], features_plus1[i+1], bias=False) for i in range(len(features_plus1)-1)])

    def forward(self, x):
        for filter in self.filters[:-1]:
            x = filter(x)
            x = F.relu(x)
        out = self.filters[-1](x)
        return out
    
def initialize_networks(feature_count, device, nn_init_num = 10):
    net_list = []
    for i in range(nn_init_num):
        net = Net([feature_count, 50,50]).to(device)
        net.train()
        net_list.append(net)
    return net_list

def model_to_params(model):
    return torch.cat([torch.flatten(filter.weight) for filter in model.filters])

def eval_model_from_params(params, dat, n_features = [5,50,50]):
    # Expects params of shape [N, n_params_flat] or [n_params_flat]
    # dat of shape [n_batch, n_features[0]]

    n_features_plus1 = n_features + [1]
    len_params = [n_features_plus1[i] * n_features_plus1[i+1] for i in range(len(n_features))]
    if len(params.shape) == 1:
        params = params[None, :] # if there is one particle, expand dmiension
    param_breakpoints = np.concatenate(([0],np.cumsum(len_params)))

    out = torch.unsqueeze(dat, 0).repeat(params.shape[0], 1, 1) # [N, n_batch, n_features[0]]

    # eval

    for i in range(len(param_breakpoints)-1):
        filter_weight = params[:, param_breakpoints[i]:param_breakpoints[i+1]]
        filter_mat = filter_weight.view(params.shape[0], n_features_plus1[i], n_features_plus1[i+1]) # should be [N, n_features, 50] then [N, 50, 1]
        out = torch.matmul(out, filter_mat) # [N, n_batch, n_features[i+1]]
        if i < len(param_breakpoints)-2:
            out = F.relu(out)

    return out.squeeze(2) # shape [N, n_batch, 1]

def criterion_from_params(params, features, y_true, n_features,n_train):
    # N = number of particles
    y_pred = eval_model_from_params(params, features, [n_features,50,50]) # [N, n_batch]
    # print(params.shape, y_pred.shape, y_true.shape)
    y_true_expanded = y_true.view(1, -1)
    # y_true_expanded = y_true_expanded.repeat(y_pred.shape[0], 1) # [N, n_batch]

    log_v_noise, v_noise = np.log(0.5), 0.5
    n_train = y_pred.shape[0]
    llhood_data = -0.5 * np.log(2*np.pi) * log_v_noise \
                      -0.5 * torch.mean((y_pred - y_true_expanded)**2 / v_noise, dim=1) # [N]
    log_prior = -0.5 * torch.sum(params**2, dim=1)/n_train

    return llhood_data, log_prior # both of shape [N]

def llhood_posterior(params, features, y_true, n_features,n_train):
    llhood_data, log_prior = criterion_from_params(params, features, y_true, n_features,n_train)
    return llhood_data + log_prior

def nllhood_posterior(params, features, y_true, n_features,n_train):
    return -llhood_posterior(params, features, y_true, n_features,n_train)

def nllhood(params, features, y_true, n_features,n_train):
    llhood, _ = criterion_from_params(params, features, y_true, n_features,n_train)
    return -llhood

def mmse_logprior(params, features, y_true, n_features, targ_std, targ_mean, train_features, train_y):
    # For evaluation only
    y_pred = eval_model_from_params(params, features, [n_features,50,50]) # [N, n_batch]
    y_true_expanded = y_true.view(1, -1)

    y_pred = y_pred * targ_std + targ_mean
    y_true_expanded = y_true_expanded * targ_std + targ_mean
    y_true = y_true * targ_std + targ_mean
    # y_true_expanded = y_true.view(1, -1)
    # y_true_expanded = y_true_expanded.repeat(y_pred.shape[0], 1) # [N, n_batch]

    train_y_pred = eval_model_from_params(params, train_features, [n_features, 50,50])
    train_y_pred = train_y_pred * targ_std + targ_mean
    train_y = train_y * targ_std + targ_mean
    neg_log_var = -torch.log(torch.mean((train_y_pred - train_y[None,:])**2))

    prob = np.sqrt(np.exp(neg_log_var.item())/(2*np.pi)) * torch.exp(-0.5 * (y_pred - y_true_expanded)**2 * torch.exp(neg_log_var))
    ll = torch.mean(torch.log(torch.mean(prob, axis=0)))

    mmse = torch.mean((y_true - torch.mean(y_pred, dim=0))**2)
    # errs = torch.min(torch.mean((y_true[None,:] - y_pred)**2,dim=1))
    # print("smallest mmse", errs)
    # print("train_y", train_y_pred[:20])
    return mmse, ll, neg_log_var # both of shape [N]

def criterion(net_list, features, y_true):
    # compute log likelihood
    log_v_noise, v_noise = np.log(0.5), 0.5
    n_train = len(net_list) # number of networks (particles)

    llhood_data_list = torch.empty(n_train)
    log_prior_list = torch.empty(n_train)
    for i, model in enumerate(net_list):
        y_pred = model(features)
        llhood_data = -0.5 * np.log(2*np.pi) * log_v_noise \
                      -0.5 * torch.mean((y_pred - y_true.view(-1, 1))**2 / v_noise, dim=1)
        log_prior = 0
        for p in model.filters:
            log_prior = log_prior + torch.sum(p.weight**2)
        
        llhood_data_list[i] = llhood_data
        log_prior_list[i] = -0.5 * log_prior


    return llhood_data_list, log_prior_list # shape [N]
