import numpy as np
import torch
import scipy.sparse as sp


#add_fake_node
def cut_edge_selection(adj, innormal_features, features, file_path,num_fake):
    

    num_ori = adj.shape[0]
    
    # num_new = num_ori + num_fake
    
    C = np.zeros((num_ori, num_fake))
    CT = np.zeros((num_fake, num_ori))
    B = np.zeros((num_fake, num_fake))
    adj = np.concatenate((adj, C), axis = 1)
    CTB = np.concatenate((CT, B), axis = 1)
    adj = np.concatenate((adj, CTB), axis = 0)
    feat_fake = node_edge_weight_computation(innormal_features,num_fake)
    features = np.concatenate((features, feat_fake), 0)
    np.save(file_path, features)
    features = torch.from_numpy(features)
    return adj, features
#gaussian_dist
def node_edge_weight_computation(innormal_features,num_fake):
    feat_mean = np.mean(innormal_features, axis = 0)
    feat_std = np.std(innormal_features, axis = 0)
    feat_fake = np.zeros((num_fake, innormal_features.shape[1]))
    for i in range(innormal_features.shape[1]):
        feat_fake[:,i] = np.random.normal(feat_mean[0, i], feat_std[0, i], num_fake).reshape(feat_fake[:,i].shape)
    feat_fake = np.where(feat_fake > 0.5, 1, 0).astype(np.float32)
    feat_fake, _ = normalize(feat_fake)
    return feat_fake

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum

