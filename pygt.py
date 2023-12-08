from torch_geometric_temporal.dataset import WikiMathsDatasetLoader, TwitterTennisDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, remove_self_loops
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pylab as pl
import torch
import os
from ot.lp import wasserstein_1d
from ot.utils import proj_simplex

import ot
from ot.gromov import gromov_wasserstein2

rng = np.random.RandomState(42)

loader = TwitterTennisDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)



#loader = WikiMathsDatasetLoader()
#dataset = loader.get_dataset(lags=14)
#train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)
device = "cuda" if torch.cuda.is_available() else "cpu"

wtf =[]
for i in train_dataset:
    wtf.append(i.edge_index)
wtf2 = []
for idx, i in enumerate(wtf):
    if idx == 0:
        pass
    else:    
        wtf2.append((wtf[i]==wtf[i-1]).sum())
##Precompute OT experiment


def min_weight_gw(C1, C2, a2, nb_iter_max=50, lr=1e-2):
    """ solve min_a GW(C1,C2,a, a2) by gradient descent"""

    a0 = rng.rand(C1.shape[0])  # random_init
    a0 /= a0.sum()  # on simplex
    a1_torch = torch.tensor(a0).requires_grad_(True)
    a2_torch = torch.tensor(a2)

    loss_iter = []

    for i in tqdm(range(nb_iter_max)):

        loss = gromov_wasserstein2(C1.to(device), C2.to(device), a1_torch.to(device), a2_torch.to(device))

        loss_iter.append(loss.clone().detach().cpu().numpy())
        loss.backward()

        #print("{:03d} | {}".format(i, loss_iter[-1]))

        # performs a step of projected gradient descent
        with torch.no_grad():
            grad = a1_torch.grad
            a1_torch -= grad * lr   # step
            a1_torch.grad.zero_()
            a1_torch.data = ot.utils.proj_simplex(a1_torch)

    a1 = a1_torch.clone().detach().cpu()

    return a1, loss_iter



#Precompute snapshot OT matrices
for data_type in ['train', 'test']:
    if data_type == 'train':
        dataset = train_dataset
    else:
        dataset = test_dataset
    prev_adj = []
    for time, snapshot in enumerate(dataset):
        #expand
        smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
        curr_adj = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
        if time == 0:
            prev_adj = curr_adj
        else:
            a1, loss_iter = min_weight_gw(prev_adj, curr_adj, ot.unif(prev_adj.shape[0]), nb_iter_max=30, lr=1e-2)
            T_est = ot.gromov_wasserstein(curr_adj.to(device), prev_adj.to(device), 
                                        torch.tensor(ot.unif(prev_adj.shape[0])).to(device), a1.to(device))
            T_est = T_est.detach().cpu().numpy()
            prev_adj = curr_adj
            np.save(f'ot_weights_dyna/{data_type}_b_{time}.npy', T_est)
        #y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)


def trunc_train_test(*args):
    files = os.listdir('ot_weights_dyna')
    files = ['ot_weights_dyna/' + i for i in files]
    snap_org = {'test':{}, 'train':{}}
    for file in files:
        if 'train' in file:
            A = np.load(file)
            bnum = int(file.split('_')[-1][:-4])
            snap_org['train'][bnum] = A
        elif 'test' in file:
            A = np.load(file)
            bnum = int(file.split('_')[-1][:-4])
            snap_org['test'][bnum] = A#.append({bnum:A})
    
    # merge with train/test data
    train_trunc = []
    val_trunc = []
    test_trunc = []
    val_cutoff = len(snap_org['test'].keys())//2

    for time, batch in enumerate(train_dataset):
        if time == 0:
            pass
        elif time <= len(snap_org['train'].keys()):
            batch.otmat = torch.tensor(snap_org['train'][time])
            train_trunc.append(batch)
        else:
            pass
    
    for time, batch in enumerate(test_dataset):
        if time == 0:
            pass
        elif time <= val_cutoff:
            batch.otmat = torch.tensor(snap_org['test'][time])
            val_trunc.append(batch)

        elif ((time > val_cutoff) & (time <= len(snap_org['test'].keys()))):
            batch.otmat = torch.tensor(snap_org['test'][time])
            test_trunc.append(batch)
        else:
            pass
    return train_trunc, val_trunc, test_trunc            

train_trunc, val_trunc, test_trunc = trunc_train_test()


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.filters = filters
        self.recurrent1 = GConvGRU(node_features, 128, 4)
        #self.recurrent2 = GConvGRU(256, 128, 2)
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent1(x, edge_index, edge_weight)
        h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        #h = F.relu(h)
        h = self.linear(h)
        return h


## Baseline model
model = RecurrentGCN(node_features=16, filters=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
full_ttrunc = val_trunc + test_trunc 

def train_baseline(train_trunc):
    model.train()
    for epoch in tqdm(range(50)):
        cost = 0
        prev_h = None
        for time, snapshot in enumerate(train_trunc):
            y_hat = model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            cost = torch.mean((y_hat - snapshot.y.to(device))**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(cost)

def eval_baseline(full_ttrunc):
    model.eval()
    cost = 0
    for time, snapshot in enumerate(full_ttrunc):
        y_hat = model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
        cost = cost + torch.mean((y_hat-snapshot.y.to(device))**2)
        del snapshot
        del y_hat
    cost = cost / (time+1)
    cost = cost.item()
    print(cost)


train_baseline(train_trunc)
torch.save(model.state_dict(), 'models/baseline.pt')
torch.cuda.empty_cache()
eval_baseline(train_trunc)
eval_baseline(val_trunc)
eval_baseline(test_trunc) 

## Node Feature Model

model = RecurrentGCN(node_features=1016, filters=2)
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
full_ttrunc = val_trunc + test_trunc 
torch.cuda.empty_cache()

def xmod_train(train_trunc):
    for epoch in tqdm(range(50)):
        cost = 0
        prev_h = None
        for time, snapshot in enumerate(train_trunc):
            mod_x= torch.hstack([snapshot.x, snapshot.otmat])
            smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
            #curr_adj = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
            y_hat = model(mod_x.to(device), smod[0].to(device), smod[1].to(device))
            cost = torch.mean((y_hat - snapshot.y.to(device))**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(cost)

def xmod_eval(full_ttrunc):
    model.eval()
    cost = 0
    for time, snapshot in enumerate(full_ttrunc):
        torch.cuda.empty_cache()
        mod_x = torch.hstack([snapshot.x, snapshot.otmat])
        smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
            #curr_adj = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
        y_hat = model(mod_x.to(device), smod[0].to(device), smod[1].to(device))
        #y_hat = model(mod_x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
        cost = cost + torch.mean((y_hat-snapshot.y.to(device))**2).detach().cpu().numpy()
        del snapshot
        del y_hat
    cost = cost / (time+1)
    cost = cost.item()
    print(cost)
    #print("MSE: {:.4f}".format(cost))

xmod_train(train_trunc)
torch.cuda.empty_cache()
torch.save(model.state_dict(), 'models/xmod.pt')
xmod_eval(train_trunc)
xmod_eval(test_trunc)
xmod_eval(val_trunc)


## Adj-weighting experiment
mod_train_adj_batches = []
for snapshot in train_trunc:
    smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
    di = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
    di = sum(di) + di*snapshot.otmat.T
    (di_ei, ei_ea) = dense_to_sparse(di)
    new_snapshot = Data(x=snapshot.x, edge_index = di_ei, edge_attr = ei_ea, y = snapshot.y)
    mod_train_adj_batches.append(new_snapshot)
mod_test_adj_batches = []
mod_val_adj_batches = []

for snapshot in test_trunc:
    smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
    di = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
    #di = to_dense_adj(edge_index = snapshot.edge_index, edge_attr=snapshot.edge_attr).squeeze(0)
    di = sum(di) + di*snapshot.otmat.T
    (di_ei, ei_ea) = dense_to_sparse(di)
    new_snapshot = Data(x=snapshot.x, edge_index = di_ei, edge_attr = ei_ea, y = snapshot.y)
    mod_test_adj_batches.append(new_snapshot)
for snapshot in val_trunc:
    smod = add_self_loops(snapshot.edge_index, snapshot.edge_attr, num_nodes=1000)
    di = to_dense_adj(edge_index = smod[0], edge_attr=smod[1]).squeeze(0)
    #di = to_dense_adj(edge_index = snapshot.edge_index, edge_attr=snapshot.edge_attr).squeeze(0)
    di = sum(di)*di*snapshot.otmat
    (di_ei, ei_ea) = dense_to_sparse(di)
    new_snapshot = Data(x=snapshot.x, edge_index = di_ei, edge_attr = ei_ea, y = snapshot.y)
    mod_val_adj_batches.append(new_snapshot)

model = RecurrentGCN(node_features=16, filters=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_baseline(mod_train_adj_batches)
torch.save(model.state_dict(), 'models/adj_mod.pt')
torch.cuda.empty_cache()
eval_baseline(mod_train_adj_batches) #0.419025
eval_baseline(mod_val_adj_batches) #0.419025
eval_baseline(mod_test_adj_batches) #0.419025


## Loss function - minimize distance between H_t^{(l)}d and H_{t-1}^{(l)}
# 1d model

class OTRecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(OTRecurrentGCN, self).__init__()
        self.filters = filters
        self.recurrent1 = GConvGRU(node_features, 128, 4)
        #self.recurrent2 = GConvGRU(256, 128, 2)
        self.linear = torch.nn.Linear(128, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent1(x, edge_index, edge_weight)
        out = F.relu(h)
        logits = torch.sqrt(torch.sum(out, axis = 1))
        #logits = torch.div(torch.sum(out, axis = 1), torch.sum(out))
        out = self.linear(out)
        return out, logits


model = OTRecurrentGCN(node_features=16, filters=2)
#model = model.to(device)
x_torch = torch.tensor(np.arange(1000, dtype=np.float64))#.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#out, hprev = model(train_trunc[0].x, train_trunc[0].edge_index, train_trunc[0].edge_attr)
#out, logits = model(train_trunc[1].x, train_trunc[1].edge_index, train_trunc[1].edge_attr)

from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
sloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


def train_baseline_lossy(train_trunc):
    model.train()
    h_prev = torch.empty(1000).uniform_(0,1)
    h_prev /= h_prev.sum()
    for epoch in tqdm(range(50)):
        logit_list = []
        h_prev_list = [] 
        cost = 0
        for time, snapshot in enumerate(train_trunc):
            y_hat, logits = model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            cost = torch.mean((y_hat - snapshot.y.to(device))**2)
            wloss = sloss(logits.unsqueeze(1), h_prev.detach().unsqueeze(1))
            logit_list.append(logits.clone().detach().numpy())
            h_prev_list.append(h_prev.clone().detach().numpy())
            #wloss = wasserstein_1d(x_torch, x_torch, logits.clone().detach(), hprev.clone().detach(), p=2)
            h_prev = logits
            cost = cost + wloss
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(cost)

#import pickle
#with open('epoch_0_logits.pickle', 'wb') as handle:
#    pickle.dump(logit_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('epoch_0_hprev.pickle', 'wb') as handle:
#    pickle.dump(h_prev_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



def eval_baseline_lossy(full_ttrunc):
    model.eval()
    cost = 0
    for time, snapshot in enumerate(full_ttrunc):
        y_hat, logits = model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
        cost = cost + torch.mean((y_hat-snapshot.y.to(device))**2)
    cost = cost / (time+1)
    cost = cost.item()
    print(cost)

train_baseline_lossy(train_trunc)
eval_baseline_lossy(train_trunc) #0.419105
eval_baseline_lossy(val_trunc) #0.419105
eval_baseline_lossy(test_trunc) #0.419105


#sloss(torch.tensor(logit_list[1]).unsqueeze(1), torch.tensor(h_prev_list[-1]).unsqueeze(1))
train_trunc[1].edge_index

ora = to_dense_adj(edge_index = train_trunc[1].edge_index, edge_attr = train_trunc[1].edge_attr)
ora = ora.numpy()
import pandas as pd
ora = pd.DataFrame(ora.squeeze(0))
ora.values[[np.arange(len(ora))]*2] = np.nan

new_ora = ora.stack().reset_index()
new_ora.columns = ['source', 'target', 'weight']
new_ora = new_ora[new_ora['weight']>0]

new_ora.to_csv('ora_viz.csv', index = False)