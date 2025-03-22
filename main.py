import warnings  
warnings.simplefilter(action='ignore', category=FutureWarning)  

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from parse import parser_add_main_args
import warnings
import model_run_GMM

# 忽略 FutureWarning  

def main(args):
    device = f'cuda:{args.gpu_select}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.cpu:
        device = torch.device('cpu')
    
    dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4, val_size=0.2)

    if args.method in ('multisage') and args.classification == 'GMM_mod':
        model_run_GMM.GMM_mod_run(args, device, dataset)
    else:
        print("wrong method")
    
    
if __name__ == "__main__":
    np.random.seed(0)
    ### Parse args ###
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)


