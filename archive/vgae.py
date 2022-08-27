
import torch
from torch.autograd.grad_mode import F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

from torch_geometric.nn import VGAE

import scipy.sparse as sp
import numpy as np
import networkx as nx

class SimuData():
    """Simulate graph data"""
    def __init__(self, n_node=10, n_graph=30):
        self.n_node = n_node
        self.n_graph = n_graph

    def simu_adj_wgh(self):
        adj_list = []

        for i in range(self.n_graph):
            ## zero matrix
            A = torch.zeros(self.n_node, self.n_node)

            # first five nodes weights: uniform(0,1)A[:5,:5] = W
            W = torch.rand(5,5)

            ## symmetric
            i, j = torch.triu_indices(5, 5)
            W[i, j] = W.T[i, j]

            A[:5,:5] = W
            adj_list.append(A)  

        return adj_list

    def simu_adj_diag(self):
        adj_list = []

        for i in range(self.n_graph):
            A = torch.eye(self.n_node)
            adj_list.append(A)  

        return adj_list

    def simu_adj_m(self):
        """generating adjacency matrix"""
        adj_wgh = self.simu_adj_wgh()
        #adj_wgh = self.simu_adj_diag()
        adj_m_list =[]
        for _, adj in enumerate(adj_wgh):
            adj[adj>0.5] = 1
            adj[adj<0.5] = 0
            adj_m_list.append(adj)    
        return adj_m_list

    def graph_dataset(self):
        dataset =[]
        simu_adj = self.simu_adj_m()

        for _, adj in enumerate(simu_adj):
            edge_index_temp = sp.coo_matrix(adj)
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index_A = torch.LongTensor(indices) 
            x = self.get_x_feature()
            data = Data(x=x, edge_index=edge_index_A)
            dataset.append(data)

        return dataset

    def get_x_feature(self):
        x = torch.arange(self.n_node)
        x_onehot = torch.eye(self.n_node)[x,:] 

        return torch.FloatTensor(x_onehot)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) 
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train():
    transform = T.Compose([
                T.NormalizeFeatures(),
                T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True,
                                split_labels=True, add_negative_train_samples=False),])
    
    simu_graph = SimuData()
    dataset = simu_graph.graph_dataset()
    n_node = simu_graph.n_node
    
    out_channels = 2
    num_features = n_node
    
    model = VGAE(Encoder(num_features, out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    model.train()
    
    for epoch in range(epochs):
        loss_total = 0
        optimizer.zero_grad()
        
        for i in range(len(dataset)):
            train_data, val_data, test_data = transform(dataset[i])
            z = model.encode(train_data.x, train_data.edge_index)
            loss = model.recon_loss(z, train_data.pos_edge_label_index)
            loss = loss + (1 /num_features) * model.kl_loss()
            loss_total += loss
            import pdb; pdb.set_trace()

        loss_avg = loss_total/len(dataset)
        loss_avg.backward()
        optimizer.step()
        
        print(loss_avg)

    import pdb; pdb.set_trace()
    


train()
    


