import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn import SAGEConv
import scipy.sparse
import numpy as np
import math
from operator import itemgetter
from torch.autograd import Variable

class MultiSAGE(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, etypes):  
        super().__init__()  
        self.edge_type_convs = nn.ModuleDict({  
            etype: SAGEConv(in_channels, hidden_channels) for etype in etypes  
        })  
        self.all_edges_conv = SAGEConv(in_channels, hidden_channels)  
        self.weight = nn.Parameter(torch.ones(len(etypes)))  
        self.final_conv = SAGEConv(hidden_channels, out_channels)  
        self.bn = nn.BatchNorm1d(hidden_channels)  
        self.dropout = dropout  
        self.reset_parameters()  

    def reset_parameters(self):  
        for conv in self.edge_type_convs.values():  
            conv.reset_parameters()  
        self.all_edges_conv.reset_parameters()  
        self.final_conv.reset_parameters()  
        self.bn.reset_parameters()  
        nn.init.ones_(self.weight)  

    def forward(self, data, feat):  
        x = feat  
        edge_outputs = []  
        all_edges_list = []  

        # Process each edge type separately  
        for etype, conv in self.edge_type_convs.items():  
            src, dst = data.edges(etype=etype)  
            edges_tensor = torch.stack((src, dst))  
            all_edges_list.append(edges_tensor)  
            edge_output = conv(x, edges_tensor)  
            edge_outputs.append(edge_output)  

        # Concatenate all edge tensors to form a single tensor for all edges  
        all_edges_tensor = torch.cat(all_edges_list, dim=1)  

        # Process all edges together  
        all_edges_output = self.all_edges_conv(x, all_edges_tensor)  

        # Combine results with learnable weights for specific edge types  
        combined_output = sum(w * out for w, out in zip(self.weight, edge_outputs))  
        combined_output += all_edges_output  # Add the all edges output directly  
        combined_output = F.relu(combined_output)  
        combined_output = self.bn(combined_output)  
        combined_output = F.dropout(combined_output, p=self.dropout, training=self.training)  

        # Final SAGE layer  
        final_output = self.final_conv(combined_output, all_edges_tensor)  
        return final_output  
     
