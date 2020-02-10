import torch
import torch.nn as nn

class Controller(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.l1 = nn.Linear(input_size, output_size, bias=True)    
        
    def forward(self, x):
        
        out = torch.tanh(self.l1(x))
        return out

    def set_weights(self, weights):
        self.l1.weight = weights
        
    def set_bias(self, bias):
        self.l1.bias = bias
        

