import os
import math

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import classification_report

import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1004)

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
if args.dataset == "sbm":
    args.num_users = 0
    args.num_info = 0

