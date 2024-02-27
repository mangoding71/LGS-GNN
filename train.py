
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import process
from components import localsmooth,graphcut
from components.gcn import GCN
from components.prediction import accuracy,recall
from threatmodel.universal_attack import universal_attack
from __future__ import division
from __future__ import print_function
import os.path as op

os.environ["CUDA_VISIBLE_DEVICES"]="0" 


parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=True,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default="citeseer", help='The dataset.')
parser.add_argument('--save_model', type=bool, default=True)
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
num_nodes = 5
i = 2
G = np.random.randint(2, size=(num_nodes, num_nodes))  
theta = np.random.rand(num_nodes, 3)  
delta_A_delete = np.random.randint(2, size=(num_nodes, num_nodes))
delta_A_add = np.random.rand(num_nodes, num_nodes)
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
num_fake = int(tmp_adj.shape[0] * fake_rate)
global new_feat
global new_adj

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


t_total = time.time()
for epoch in range(epochs):
    train(epoch)

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.save_model:
    model_filename = f'models/{args.dataset}_gnn.pth'
    state_dict_filename = f'models/{args.dataset}_gnn.pkl'
    
    torch.save(model, model_filename)
    torch.save(model.state_dict(), state_dict_filename)


def test(feat, adj_m):
    model.eval()
    output = model(feat, adj_m)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output

ori_output = test(features, adj)
correct_res = ori_output[idx_train, labels[idx_train]] 


train_foolrate = []
                                   
for i in range(0,10):
    folder_path = op.join("./", "step{3}_new_adj/{0}/fake{1}_radius{2}".format(args.dataset, num_fake, radius, step))
    
    os.makedirs(folder_path, exist_ok=True)
    if not op.exists(folder_path):
        os.mkdir(folder_path)
        
    adj_path = op.join(folder_path, 'adj{}.npy'.format(i))
    feat_path = op.join(folder_path, 'feat{}.npy'.format(i))
    new_adj, new_feat = graphcut.cut_edge_selection(tmp_adj, tmp_feat, feat, feat_path,num_fake)
    new_feat = new_feat.cuda()
    G_smooth=localsmooth.graph_smoothing(G, delta_A_add)
    f_local_avg = localsmooth.local_averaging(G_smooth, theta)

    train_attack_coverage = universal_attack(i, 50, adj_path, model, tmp_adj, num_fake, new_adj
                     , idx_train, new_feat, ori_output, num_classes, step,labels) #train_attack_coverage = universal_attack(i, 50, adj_path)
    train_foolrate.append(train_attack_coverage)

print ('the final train attack coverage', train_foolrate)





