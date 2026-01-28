from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.make_dependent = configs.make_dependent
        self.channels = configs.n_features
        
        self.use_norm = configs.use_norm


        self.Linear = nn.Linear(self.seq_len, 1)
        self.Linear.weight = nn.Parameter((1/self.seq_len) * torch.ones([1, self.seq_len]))
        self.project = nn.Linear(self.channels, 1)
            
    
    def forecast(self, x):
        B, S, C = x.shape # [batch, seq_len, variates]
        x = x.permute(0, 2, 1)

        # Apply linear layers
        output = self.Linear(x)  # [Batch, Channel, Pred Len * Channels]

        if self.make_dependent:
            output = output.view(output.shape[0], C, self.pred_len, C)
            output = output[:, :, :, 0]
            
        output = output.permute(0, 2, 1)
    
    def regression(self, x):
        B, S, C = x.shape # [batch, seq_len, variates]
        x = x.permute(0, 2, 1)
        output = self.Linear(x).squeeze(-1)
        output = self.project(output).squeeze(-1)
        return output
    
    def forward(self, x, x_mark, dec_inp, y_mark, batch_general):
        #x = torch.cat([x, batch_general], dim = -1) if batch_general else x
        #print(x.shape)
        output = self.regression(x)
        return output
