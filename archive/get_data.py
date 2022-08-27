from os import walk
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def norm_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

def get_filename(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return f

def compute_cent(mat):
    return (np.sum(mat, axis=0))



class LoadData():
    """Load graph data"""
    def __init__(self, file_path, samples, non_orphan_list=None, n_node=38):
        self.samples = samples
        self.file_path = file_path
        self.nSample = len(samples)
        self.non_orphan_list = non_orphan_list
        self.n_node = n_node
        self.n_connect = len(non_orphan_list)

    def get_adj_wgh(self):
        """generating weighted adjacency matrix"""
        adj_orig_list =[]
        for sample in self.samples:
            f_name = self.file_path + "/"+ sample
            adj = np.asarray(pd.read_csv(f_name, index_col = 0, iterator = False))
            adj_orig_list.append(adj)    
        return adj_orig_list

    def get_adj_m(self):
        """generating the 0-1 adjacency matrix without diagonal elements"""
        adj_wgh = self.get_adj_wgh()
        adj_m_list =[]
        for _, adj in enumerate(adj_wgh):
            adj[adj > 0] = 1
            for i in range(self.n_node):
                adj[i, i] = 0
            for i in range(self.n_node):
                for j in range(self.n_node):
                    if adj[i, j] == 1:
                        adj[j, i] = 1
            for i in range(self.n_node - 1, -1, -1):
                if i not in self.non_orphan_list:
                    adj = np.delete(adj, (i), axis=0)
                    adj = np.delete(adj, (i), axis=1)
            adj_m_list.append(adj)    
        return adj_m_list
    
    def adj_without(self):
        adj_wgh = self.get_adj_wgh()
        sample = np.eye(self.n_connect)
        adj_without = np.array([sample])
        for _, adj in enumerate(adj_wgh):
            adj[adj > 0] = 1
            for i in range(self.n_node):
                adj[i, i] = 0
            for i in range(self.n_node):
                for j in range(self.n_node):
                    if adj[i, j] == 1:
                        adj[j, i] = 1
            for i in range(self.n_node - 1, -1, -1):
                if i not in self.non_orphan_list:
                    adj = np.delete(adj, (i), axis=0)
                    adj = np.delete(adj, (i), axis=1)
            adj = adj.astype(np.float)
            adj_without = np.append(adj_without, [adj], axis=0)
        adj_without = np.delete(adj_without, 0, axis=0)
        return adj_without

    def get_adj_m_t(self):
        """0-1 adjacency matrix without diagonal elements tensor format"""
        adj_wgh = self.get_adj_wgh()
        adj_m_list =[]
        for _, adj in enumerate(adj_wgh):
            adj[adj > 0] = 1
            for i in range(self.n_node):
                adj[i, i] = 0
            for i in range(self.n_node):
                for j in range(self.n_node):
                    if adj[i, j] == 1:
                        adj[j, i] = 1
            for i in range(self.n_node - 1, -1, -1):
                if i not in self.non_orphan_list:
                    adj = np.delete(adj, (i), axis=0)
                    adj = np.delete(adj, (i), axis=1)
            adj = sparse_to_tuple(sp.coo_matrix(adj))
            adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T), 
                            torch.FloatTensor(adj[1]), 
                            torch.Size(adj[2]))
            adj_m_list.append(adj)    
        return adj_m_list
    
    def get_adj_label(self):
        """0-1 adjancency matrix with diagonal elements tensor format"""
        adj_m = self.get_adj_m()
        adj_label_list =[]
        
        for _, adj in enumerate(adj_m):
            adj_label = adj + sp.eye(adj.shape[0])
            adj_label = sparse_to_tuple(sp.coo_matrix(adj_label))
            #adj_label = sparse_to_tuple(adj_label)
            adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
            adj_label_list.append(adj_label)
        return adj_label_list

    def adj_with(self):
        adj_m = self.get_adj_m()
        sample = np.eye(self.n_connect)
        adj_with = np.array([sample])
        for _, adj in enumerate(adj_m):
            adj_label = adj + sp.eye(adj.shape[0])
            adj_label = adj_label.astype(np.float)
            adj_with = np.append(adj_with, [adj_label], axis=0)
        adj_with = np.delete(adj_with, 0, axis=0)
        return adj_with

    def get_adj_norm(self):
        """0-1 normalized adjancency matrix tensor format"""
        adj_m = self.get_adj_m()
        adj_norm_list =[]

        for _, adj in enumerate(adj_m):
            adj_norm = preprocess_graph(adj)
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
            adj_norm_list.append(adj_norm)
        return adj_norm_list

    def adj_norm(self):
        adj_m = self.get_adj_m()
        sample = np.eye(self.n_connect)
        adj_norm = np.array([sample])
        for _, adj in enumerate(adj_m):
            adj_norm_ = norm_graph(adj)
            adj_norm_ = adj_norm_.astype(np.float)
            adj_norm = np.append(adj_norm, [adj_norm_], axis=0)
        adj_norm = np.delete(adj_norm, 0, axis=0)
        return adj_norm

    def get_feature(self):
        """generating feature matrix X tensor format"""
        adj_wgh = self.get_adj_wgh()
        x_list = []
        for _, adj in enumerate(adj_wgh):
            x_feature  = adj
            x_feature  = csr_matrix(x_feature)
            x_feature  = sparse_to_tuple(x_feature)
            x_feature  = torch.sparse.FloatTensor(torch.LongTensor(x_feature[0].T), 
                            torch.FloatTensor(x_feature[1]), 
                            torch.Size(x_feature[2]))
            x_list.append(x_feature)
        return x_list

    def get_x_feature(self):
        """one-hot encoding tensor format"""
        x_onehot = torch.eye(len(self.non_orphan_list))
        return torch.FloatTensor(x_onehot)

    def one_hot(self):
        sample = np.eye(self.n_connect)
        feature = np.array([sample])
        for i in range(self.nSample):
            feature = np.append(feature,[np.eye(self.n_connect)],axis=0)
        feature = np.delete(feature, 0, axis=0)
        return feature





