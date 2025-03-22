from model import *
import torch
import torch.nn as nn
import scipy.sparse as sp
from collections import defaultdict
import numpy as np
import dgl
import torch
import os
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from scipy.io import loadmat
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.optim.lr_scheduler import MultiStepLR


def parse_method(args, dataset, n, c, d, train_pos, device): 
    if args.method == 'multisage':
        model = MultiSAGE(in_channels=d, hidden_channels=args.hidden_channels, 
                     out_channels=c, dropout=args.dropout, etypes = dataset[0].etypes).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--method', '-m', type=str, default='multisage')
    parser.add_argument('--classification', type=str, default='GMM_mod',
                        choices=['GMM_mod'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--Mahalanobis', type=str, default='diagonal_nequal', 
                        choices=['base','diagonal_nequal', 'diagonal_equal', 'matrix'],
                        help='choose Mahalanobis types')
    parser.add_argument('--early_stop', type=bool, default=True,
                        help='early_stop')
    parser.add_argument('--gmm_K', type=int, default=3,
                        help='select k for gmm or multi center')
    parser.add_argument('--Measurement', type=str, default='mixed',
                        choices=['closest', 'mixed'],
                        help='choose Measurement types')
    parser.add_argument('--gpu_select', type=int, default=0,
                        help='select cuda from 0-7')

