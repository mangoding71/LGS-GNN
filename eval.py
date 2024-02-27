import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import os.path as op
from utils import process
from components.prediction import accuracy,recall,evaluate_model
from __future__ import division
from __future__ import print_function

from components.gcn import GCN
os.environ["CUDA_VISIBLE_DEVICES"]="0" 




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,  help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default="citeseer",   help='The name of the network dataset.')
parser.add_argument('--save_model', type=bool, default=False)
args = parser.parse_args()
seed=42
epochs=200  #epochs=200
lr=0.01
weight_decay=5e-4
hidden=16
dropout=0.5
radius=12
fake_rate=0.02
step=10
sample_percent=40



np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)


if args.dataset == "polblogs":
    tmp_adj, tmp_feat, labels, train_idx, val_idx, test_idx = process.load_polblogs_data()
else:
    _, _, labels, train_idx, val_idx, test_idx, tmp_adj, tmp_feat  = process.load_data(args.dataset)

num_classes = labels.max().item() + 1

adj = tmp_adj
adj = np.eye(tmp_adj.shape[0]) + adj
adj, _ = process.normalize(adj)
adj = torch.from_numpy(adj.astype(np.float32))
feat, _ = process.normalize(tmp_feat)
feat = torch.FloatTensor(np.array(feat.todense()))
tmp_feat = tmp_feat.todense()




# Model and optimizer
model = GCN(nfeat=feat.shape[1],
            nhid=hidden,
            nclass=num_classes,
            dropout=dropout
           )
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)



if args.cuda:
    model.cuda()
    features = feat.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = train_idx.cuda()
    idx_val = val_idx.cuda()
    idx_test = test_idx.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    x = Variable(adj, requires_grad=True)
    output = model(features, x)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(adj_m):
    model.eval()
    output = model(features, adj_m)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output

t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.save_model:
    model_filename = f'models/{args.dataset}_gnn.pth'
    state_dict_filename = f'models/{args.dataset}_gnn.pkl'
    
    torch.save(model, model_filename)
    torch.save(model.state_dict(), state_dict_filename)

ori_output = test(adj)
correct_res = ori_output[idx_train, labels[idx_train]] 
num_fake = int(tmp_adj.shape[0] * fake_rate)

new_pred = []
for i in range(num_classes):
    new_pred.append(0)
for k in idx_test:
    new_pred[int(torch.argmax(ori_output[k]))] += 1

attack_coverage = []
new_acc = []   
for i in range(10):
    folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, radius, step))
    adj_path = op.join(folder_path, 'adj{}.npy'.format(i))        
    feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
    new_adj = np.load(adj_path)
    new_feat = np.load(feat_path)
    new_feat = torch.from_numpy(new_feat).float()
    new_feat = new_feat.cuda()
        
    res, acc, new_pred = evaluate_model(new_adj, new_feat, tmp_adj, num_fake, num_classes, idx_test, model, labels, ori_output)

    attack_coverage.append(res)      
    new_acc.append(acc)
    print ('the attack coverage are', attack_coverage)
    print ('the new accuracy is', new_acc)
    print ('the average attack coverage over 10 times of test is', sum(attack_coverage)/float(len(attack_coverage)))
    print ('the average new accuracy over 10 times of test is', sum(new_acc)/float(len(new_acc)))





