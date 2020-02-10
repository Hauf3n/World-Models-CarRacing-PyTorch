import pickle
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from MDN import MDN

class MDN_RNN_DataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        
        f = open(self.data_dir+f'\\{index}', "rb")
        data = pickle.load(f)
        
        data_mu = torch.tensor(data["mu"], dtype=torch.float)
        data_logvar = torch.tensor(data["logvar"], dtype=torch.float)
        eps = torch.randn_like(data_logvar)
        data_obs = data_mu + eps * torch.exp(0.5*data_logvar)
        
        data_actions = torch.tensor(data["actions"], dtype=torch.float)[0]

        #cat obs and actions as input for mdn rnn
        data = torch.cat((data_obs,data_actions), dim=1)
        seq_len = data.shape[0]

        # zero pad to a seq len of 1000
        zero_pad = torch.zeros(1000-seq_len,35)
        data = torch.cat((data,zero_pad), dim=0)
        f.close()
        return data
    
class MDN_RNN(nn.Module):

    def __init__(self, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        super(MDN_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x, state=None):
        
        y = None
        if state is None:
            y, state = self.lstm(x)
        else:
            y, state = self.lstm(x, state)
        
        pi, sigma, mu = self.mdn(y)
        
        return pi, sigma, mu, state
            
    def forward_lstm(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        if state is None:
            y, state = self.lstm(x)
        else:
            y, state = self.lstm(x, state)

        return y, state

    def loss(self, y, pi, mu, sigma):
        loss = self.mdn.loss(y, pi, mu, sigma)
        return loss

    def get_hidden_size(self):
        return self.hidden_size

