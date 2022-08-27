import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from scipy.stats import ttest_ind, levene
import scipy.stats as stats
from scipy.sparse import csr_matrix
from sklearn.mixture import GaussianMixture

### change working directory ###

os.getcwd()
os.chdir("C:\\Users\\Gang Xu\\Desktop\\VGAE")
os.getcwd()

from model import *
from get_data import *
from loss import *
import args

### set parameters ###
ipf_sample_size = 32
control_sample_size = 28

###################################################################################
######### data analysis for all_collagens_ipf_data ##########
###################################################################################

path_1 = './result_all_raw/Collagens/IPF'  ## all_collagens_ipf ##
path_2 = './result_all_raw/Collagens/Control'  ## all_collagens_control ##
path_3 = './result_all_raw/EGF/IPF'  ## all_egf_ipf ##
path_4 = './result_all_raw/EGF/Control'  ## all_egf_control ##
path_5 = './result_all_raw/Matrix/IPF'  ## all_matrix_ipf ##
path_6 = './result_all_raw/Matrix/Control'  ## all_matrix_control ##
path_7 = './result_sig_raw/Collagens/IPF'  ## sig_collagens_ipf ##
path_8 = './result_sig_raw/Collagens/Control'  ## sig_collagens_control ##
path_9 = './result_sig_raw/EGF/IPF'  ## sig_egf_ipf ##
path_10 = './result_sig_raw/EGF/Control'  ## sig_egf_control ##
path_11 = './result_sig_raw/Matrix/IPF'  ## sig_matrix_ipf ##
path_12 = './result_sig_raw/Matrix/Control'  ## sig_matrix_control ##

### specific parameters ###

path = path_3
sample_size = ipf_sample_size
train_size = sample_size
test_size = 0

data_full = LoadData(file_path=path, samples=get_filename(path), non_orphan_list=range(args.num_node), n_node=args.num_node)
cen_full = []
cen = []

for i in range(sample_size):
    cen_full.append(compute_cent(data_full.get_adj_m()[i]))
non_orphan = np.argwhere(np.sum(cen_full, axis=0)!=0).flatten().tolist()
data = LoadData(file_path=path, samples=get_filename(path), non_orphan_list=non_orphan, n_node=args.num_node)
for i in range(sample_size):
    cen.append(compute_cent(data.get_adj_m()[i]))

### construct data loader ###
adj_without = torch.from_numpy(data.adj_without())
adj_with = torch.from_numpy(data.adj_with())
adj_norm = torch.from_numpy(data.adj_norm())
one_hot = torch.from_numpy(data.one_hot())
dataset = TensorDataset(adj_without, adj_with, adj_norm, one_hot)
loader = DataLoader(dataset = dataset,batch_size = train_size)


### train model for ipf data###

def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate*(0.3**(epoch//100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model = VGAE()
optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss_value = []
acc_value = []
lik_value = []

for epoch in range(args.num_epoch):
    t = time.time()
    adjust_learning_rate(optimizer, epoch)
    model.train()
    for adj_without, adj_with, adj_norm, features in loader:

        A_pred = model(adj_norm, features, sample_size)
        mean = model.mean.reshape(adj_with.size(1), args.hidden2_dim)
        logstd = model.logstd.reshape(adj_with.size(1), args.hidden2_dim)
        loss = loss_function(args.w1,args.w2,args.w,args.norm1,args.norm2,A_pred,mean,logstd,adj_without,adj_with)
        train_acc = get_acc(A_pred,adj_with)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
        acc_value.append(train_acc.item())

    if epoch % 5 ==0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), 
              "train_acc=", "{:.5f}".format(train_acc),"time=", "{:.5f}".format(time.time() - t))






