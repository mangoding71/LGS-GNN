import torch

import numpy as np
from utils import process




def add_perturb(input_adj, idx, perturb):
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb

    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj


    for i in range(input_adj.shape[0]):      
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj


def evaluate_model(new_adj, new_feat, tmp_adj, num_fake, num_classes, idx_test, model, labels, ori_output):
    res = []
    v1 = np.zeros(tmp_adj.shape[0]).astype(np.float32)
    v2 = np.ones(num_fake).astype(np.float32)
    perturb = np.concatenate((v1, v2))
    new_pred = []
    all_acc = []
    for i in range(num_classes):
        new_pred.append(0)
    for k in idx_test:

        innormal_x_p = add_perturb(new_adj, k, perturb)
        x_p, degree_p = process.normalize(innormal_x_p + np.eye(new_adj.shape[0]))
        x_p = torch.from_numpy(x_p.astype(np.float32))
        x_p = x_p.cuda()
        output = model(new_feat, x_p)
        new_pred[int(torch.argmax(output[k]))] += 1
        test_acc = accuracy(output[idx_test], labels[idx_test])
        all_acc.append(test_acc)
        if int(torch.argmax(output[k])) == int(torch.argmax(ori_output[k])):
            res.append(0)
            print ('node {} attack failed'.format(k))
        else:
            res.append(1)
            print ('node {} attack succeed'.format(k))
    
    fooling_rate = float(sum(res)/len(res))
    overal_acc = float(sum(all_acc)/len(all_acc))
    print ('the current fooling rate is', fooling_rate)
    return fooling_rate, overal_acc, new_pred





def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    true_positives = (preds.eq(1) & labels.eq(1)).sum().double()
    actual_positives = labels.eq(1).sum().double()
    
    if actual_positives == 0:
        return 0.0
    recall = true_positives / actual_positives
    return recall
