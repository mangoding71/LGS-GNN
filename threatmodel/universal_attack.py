import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum

def universal_attack(attack_epoch, max_epoch, file_path, model, tmp_adj, num_fake, new_adj
                     , idx_train, new_feat, ori_output, num_classes, args_step,labels):
    model.eval()
    fooling_rate = 0.0
    max_iter_df = 10

    
    v1 = np.zeros(tmp_adj.shape[0]).astype(np.float32)
    v2 = np.ones(num_fake).astype(np.float32)
    v = np.concatenate((v1, v2))
    cur_foolingrate = 0.0
    epoch = 0
    results = []
    

    tmp_new_adj = np.copy(new_adj)
    # print ('the new adj', np.sum(tmp_new_adj[-num_fake:, :-num_fake]))
    while epoch < max_epoch:

        epoch += 1
        train_idx = idx_train.cpu().numpy()
        
    
        np.random.shuffle(train_idx)
        cut_time = time.time()
        for k in train_idx:
            innormal_x_p = add_anomalous_Node(tmp_new_adj, k, v)
            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_new_adj.shape[0]))  #A' = A + I
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()
            output = model(new_feat, x_p)

            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                dr, iter = IGP(innormal_x_p, x_p, k, num_classes, degree_p, model , new_feat,num_fake, args_step,train_idx,labels)
                if iter < max_iter_df-1:
                    tmp_new_adj = modify_adj(tmp_new_adj, dr, k,num_fake)
                else:
                    print ('No need to cut subgraphs')
            else:   
                print ('Node {} cutting successful'.format(k))
                #print ('Node cutting successful')
        print ('The time of cutting subplots cost is', time.time()-cut_time)
        

        res = []
        tmp_new_adj = np.where(tmp_new_adj>0.5, 1, 0)
        # print ('C adjacency matrix', np.sum(tmp_new_adj[-num_fake:, :-num_fake]))
        # print ('B adjacency matrix', np.sum(tmp_new_adj[-num_fake:, -num_fake:]))
        for k in train_idx:
            print ('Updated information for Node {}'.format(k))
            #print ('Test node', k)
            innormal_x_p = add_anomalous_Node(tmp_new_adj, k, v)            
            
            x_p, degree_p = normalize(innormal_x_p + np.eye(tmp_new_adj.shape[0]))
            x_p = torch.from_numpy(x_p.astype(np.float32))
            x_p = x_p.cuda()
            output = model(new_feat, x_p)
            if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
                res.append(0)
            else:
                res.append(1)
        fooling_rate = float(sum(res)/len(res))
        print ('the current train fooling rates are', fooling_rate)

        if fooling_rate > cur_foolingrate:
            
            cur_foolingrate = fooling_rate
            np.save(file_path, tmp_new_adj)
            
        results.append(fooling_rate)

        
    return cur_foolingrate
      
def IGP(innormal_adj, ori_adj, idx, num_classes, degree, model, new_feat,num_fake,args_step,train_idx,labels,overshoot=0.02, max_iter=30):
    model.eval()
    pred = model(new_feat, ori_adj)[idx]
    pred = pred.detach().cpu().numpy()
    I = pred.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]    
    f_i = np.array(pred).flatten()
    k_i = int(np.argmax(f_i))  
    w = np.zeros(ori_adj.shape[0])
    r_tot = np.zeros((ori_adj.size(0), ori_adj.size(0)))
    
    pert_adj = ori_adj.detach().cpu().numpy()
    pert_adj_tensor = ori_adj
    loop_i = 0
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        gradients = calculate_grad_class(pert_adj_tensor, idx, I,model,new_feat)
        
        for i in range(1, num_classes):
            w_k = gradients[i, :] - gradients[0, :]
            f_k = f_i[I[i]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
            r_i =  pert * w / np.linalg.norm(w)
            r_tot[idx, -num_fake:] = r_tot[idx, -num_fake:] + r_i[-num_fake:]
            r_tot[-num_fake:, idx] = r_tot[idx, -num_fake:]
            
        pert_adj_k = normalize_add_anomalous_Node(pert_adj, r_tot, True, idx, (1+overshoot))
            
        pert_adj_k = np.clip(pert_adj_k, 0, 1)
        pert_adj_k = torch.from_numpy(pert_adj_k.astype(np.float32))
        pert_adj_k = pert_adj_k.cuda()
        grad = calculate_grad(pert_adj_k, idx,model,new_feat,train_idx,labels,num_fake)
        r_tot -= grad*args_step
        pert_adj = normalize_add_anomalous_Node(pert_adj, r_tot, False, idx, (1+overshoot))
        pert_adj = np.clip(pert_adj, 0, 1)
        pert_adj_tensor = torch.from_numpy(pert_adj.astype(np.float32))
        pert_adj_tensor = pert_adj_tensor.cuda()
        f_i = np.array(model(new_feat, pert_adj_tensor)[idx].detach().cpu().numpy()).flatten()
        k_i = int(np.argmax(f_i))
        
        loop_i += 1
    r_tot[:, -num_fake:] = np.multiply(r_tot[:, -num_fake:].transpose(), degree).transpose()
    r_tot[-num_fake:, :-num_fake] = np.multiply(r_tot[-num_fake:, :-num_fake], degree[:-num_fake])
    
    r_tot[idx:] = r_tot[idx:] * (1 + overshoot)
    r_tot[:,idx] = r_tot[:,idx] * (1 + overshoot)

    return r_tot, loop_i

def add_anomalous_Node(input_adj, idx, perturb):
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj

    for i in range(input_adj.shape[0]):   
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj

def calculate_grad_class(pert_adj, idx, classes,model,new_feat):
    x = Variable(pert_adj, requires_grad=True)
    output = model(new_feat, x)
    grad = []
    for i in classes:
        cls = torch.LongTensor(np.array(i).reshape(1)).cuda()
        loss = F.nll_loss(output[idx:idx+1], cls) 
        loss.backward(retain_graph=True)
        grad.append(x.grad[idx].cpu().numpy())
    return np.array(grad)   

def calculate_grad(pert_adj, idx,model,new_feat,train_idx,labels,num_fake): ######exclude idx?
    x = Variable(pert_adj, requires_grad=True)
    output = model(new_feat, x)
    # ex_idx_train = train_idx.numpy()
    ex_idx_train = train_idx
    
    ex_idx_train = np.delete(ex_idx_train, np.where(ex_idx_train == idx))
    ex_idx_train = torch.LongTensor(ex_idx_train).cuda()
    loss_train = F.nll_loss(output[ex_idx_train], labels[ex_idx_train])
    loss_train.backward(retain_graph=True)
    gradient = np.array(x.grad.cpu().numpy())
    gradient[idx] = 0
    gradient[:,idx] = 0
    gradient[:-num_fake,:-num_fake] = 0

    np.fill_diagonal(gradient, np.float32(0)) 
    gradient = (gradient + gradient.transpose())/2
    return gradient

def normalize_add_anomalous_Node(ori_adj, pert, single_node, idx, rate):
    if single_node:
        a = ori_adj
        a[idx] += pert[idx] * rate
        a[:,idx] += pert[:,idx] * rate
    else:
        pert[idx] = pert[idx] * rate
        pert[:, idx] = pert[:, idx] * rate
        a = ori_adj + pert
    inv_d = 1 + np.sum(pert, 1)
    inv_d = 1.0/inv_d
    ori_adj = np.multiply(a.transpose(), inv_d).transpose()
    
    return ori_adj

def modify_adj(input_adj, perturb, idx,num_fake):
    input_adj = np.add(input_adj,perturb, casting ="unsafe")
    input_adj[idx, -num_fake:] = 1 - input_adj[idx, -num_fake:]
    input_adj[-num_fake:, idx] = input_adj[idx, -num_fake:]
    for i in range(-num_fake, 0):
        input_adj[i,i] = 0
        input_adj[:, i] = proj_lp(input_adj[:, i])
        input_adj[i] = input_adj[:, i]
        
    input_adj = np.clip(input_adj, 0, 1)
    return input_adj

def proj_lp(v, xi=12, p=2):
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(order='F')))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        v = v
    v = np.clip(v, 0, 1)
    return v
