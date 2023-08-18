import os
import math

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import classification_report

import argparse
import random
import pickle

from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()

parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1004)
parser.add_argument('--gru-hidden', type=int, default=3)
parser.add_argument('--gru-layers', type=int, default=1)
parser.add_argument('--info-dim', type=int, default=2)
parser.add_argument('--gru-input', type=int, default=2)

args = parser.parse_args()
args.dir_root = os.getcwd()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.cpu or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

print("Device Used : ", device)
args.device = device

args.dataset = "mooc"
# args.dataset = "sbm"

#ACT - MOOC
# Num Users : 7,047
# Num Targets : 97
# Num Actions 411,749
# Number of Positive Action Labels : 4,066
# Timestamp - Seconds

if args.dataset == "mooc":
    args.num_users = 7047
    args.num_info = 97

    with open('datasets/act-mooc/mooc_action_anom.pickle', 'rb') as handle:
        ret_data = pickle.load(handle)
    
    static_feats = ret_data[0]
    static_labels = ret_data[1]
    online_feats = ret_data[2]
    online_labels = ret_data[3]

if args.dataset == "sbm":
    args.num_users = 0
    args.num_info = 0


# node_gru = nn.GRU(2, args.gru_hidden, args.gru_layers)

# pytorch_total_params = sum(p.numel() for p in node_gru.parameters() if p.requires_grad)
# print("Model Parameter Count :", pytorch_total_params)

class onlineAnom(nn.Module):
    def __init__(self):
        super().__init__()

        self.GRU_LIST = nn.ModuleList([nn.GRU(args.gru_input, args.gru_hidden, args.gru_layers) for _ in range(args.num_users)])
        self.GRU_HID = [torch.randn(1,args.gru_hidden).to(device) for _ in range(args.num_users)]
        self.GRU_OUT = [0 for _ in range(args.num_users)]
        
        self.TARG_LIST = [torch.randn(args.info_dim).to(device) for _ in range(args.num_users)]

        self.agg = nn.Linear(args.info_dim + 1, args.gru_input)
        self.info_up = nn.Linear(args.info_dim + 1, args.info_dim)

        self.cls = nn.Linear(args.gru_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, target, delt):
        # print(delt)

        conc_temp = torch.concat((delt.reshape(1), self.TARG_LIST[target])).to(device)
        # print(type(conc_temp))

        agg_up = self.agg(conc_temp)
        gru_out, gru_hid = self.GRU_LIST[user](torch.unsqueeze(agg_up,0), self.GRU_HID[user])
        self.GRU_HID[user] = gru_hid
        self.GRU_OUT[user] = gru_out

        conc_retemp = torch.concat((delt.reshape(1), self.TARG_LIST[target])).to(device)

        info_up = self.info_up(conc_retemp)
        self.TARG_LIST[target] = info_up

        classi = self.cls(gru_out)

        return self.sigmoid(classi)


model = onlineAnom().to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Parameter Count :", pytorch_total_params)


class temp_dataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels, data_type="dataset"):
        self._feat_list = feats
        self._label_list = labels

    def __len__(self):
        return self._label_list.shape[0]
    
    def __getitem__(self, index):
        return (self._feat_list[:, index], self._label_list[index])


static_data = temp_dataset(static_feats, static_labels)
online_data = temp_dataset(online_feats, online_labels)

static_loader = torch.utils.data.DataLoader(static_data, batch_size=1, shuffle=False)
online_loader = torch.utils.data.DataLoader(online_data, batch_size=1, shuffle=False)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

model.train()

for idx, batch in enumerate(tqdm(static_loader)):
    corrupted = [1119, 1121, 1131, 1133, 1139, 1151, 1153, 3404, 3406]
    if idx in corrupted:
        print(batch)
        continue
    

    feats = batch[0].squeeze()
    label = torch.tensor(batch[1].item(), dtype=torch.float32).to(device)

    # print(feats)
    # print(label)

    #user, target, delta / label

    user = int(feats[0].item())
    target = int(feats[1].item())
    delta = feats[2].type(torch.FloatTensor).to(device)
    # delta = torch.tensor(feats[2], dtype=torch.float32).to(device)

    # if target == 1:
    #     print("Target 1")
    #     continue

    optimizer.zero_grad()

    model_out = model(user, target, delta).squeeze()

    # print("MODEL_OUT", model_out)
    # print("LABEL", label)

    loss = criterion(model_out, label.squeeze())
    loss.backward(retain_graph=True)

    nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()
    scheduler.step()

model.eval()

cur_lab = 0

for idx, batch in enumerate(tqdm(online_loader)):
    feats = batch[0].squeeze()
    label = int(batch[1].item())

    user = int(feats[0].item())
    target = int(feats[1].item())
    delta = feats[2].type(torch.FloatTensor).to(device)

    model_out = model(user, target, delta).squeeze()

    if cur_lab == 1 and label == 0:
        dbscan = DBSCAN(3, 7)

        gru_out_cp = model.GRU_OUT
        dbscan_inp = []

        for out in gru_out_cp:
            if out == 0:
                dbscan.append(np.zeros(args.gru_hidden))
            else:
                dbscan.append(out.numpy())

        dbscan.fit(model.GRU_OUT)

        labels = dbscan.labels_
        outliers = np.where(labels==-1)[0]

        print(outliers)