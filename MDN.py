import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MDN(nn.Module):
    
    def __init__(self, input_size, output_size, K, units=512):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.K = K
        
        self.l1 = nn.Linear(input_size, 3 * K * output_size)
        
        self.oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)
        
    def forward(self, x):
        batch_size, seq_len = x.shape[0],x.shape[1]

        out = self.l1(x)
        pi, sigma, mu  = torch.split(out, (self.K * self.output_size , self.K * self.output_size, self.K * self.output_size), dim=2)
        
         
        sigma = sigma.view(batch_size, seq_len, self.K, self.output_size)
        sigma = torch.exp(sigma)
        
        mu = mu.view(batch_size, seq_len, self.K, self.output_size)

        pi = pi.view(batch_size, seq_len, self.K, self.output_size)
        pi = F.softmax(pi, dim=2)
        
        return pi, sigma, mu
    
    def gaussian_distribution(self, y, mu, sigma):
        y = y.unsqueeze(2).expand_as(sigma)
        
        out = (y - mu) / sigma
        out = -0.5 * (out * out)
        out = (torch.exp(out) / sigma) * self.oneDivSqrtTwoPI

        return out
    
    def loss(self, y, pi, mu, sigma):

        out = self.gaussian_distribution(y, mu, sigma)
        out = out * pi
        out = torch.sum(out, dim=2)
        
        # kill (inf) nan loss
        out[out <= float(1e-24)] = 1
        
        out = -torch.log(out)
        out = torch.mean(out)
        
        return out

