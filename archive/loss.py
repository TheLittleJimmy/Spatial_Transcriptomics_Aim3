import torch
import torch.nn.functional as F

def loss_function (w1,w2,w,norm1,norm2,A_pred,mean,logstd,adj_without,adj_with):
    n = adj_without.size(0)
    N = adj_without.size(1)
    loss = 0
    for i in range(n):
        pos_weight = w1/float(adj_without[i].sum())
        neg_weight = w2/float(N**2 - adj_without[i].sum())
        weight_one_index = adj_with[i].view(-1) == 1
        weight_zero_index = adj_with[i].view(-1) == 0
        weight_tensor = torch.ones(weight_one_index.size(0))
        weight_tensor[weight_one_index] = pos_weight
        weight_tensor[weight_zero_index] = neg_weight
        loss += norm1*F.binary_cross_entropy(A_pred[i].view(-1), adj_with[i].view(-1), weight=weight_tensor)
        
    kl_divergence = w* norm2*0.5 * (1 + 2*logstd - mean**2 - torch.exp(logstd)**2).sum()
    loss -= kl_divergence
    
    return loss











