import pickle
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class VAE_DataSet(torch.utils.data.Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        f = open(self.data_dir+f'\\{index}', "rb")
        data = pickle.load(f)
        data["obs"] = torch.tensor(data["obs"]/255, dtype=torch.float).permute(0,3,1,2)
        f.close()
        return data

class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.device = None
        
        # encoder
        self.enc_conv1 = nn.Conv2d(3,32,kernel_size=4,stride=2, padding=0)
        self.enc_conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2, padding=0)
        self.enc_conv3 = nn.Conv2d(64,128,kernel_size=4,stride=2, padding=0)
        self.enc_conv4 = nn.Conv2d(128,256,kernel_size=4,stride=2, padding=0)
        
        # z
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        
        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(latent_size, 128, kernel_size=5, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        
        return out, mu, logvar   
        
    def encode(self, x):
        batch_size = x.shape[0]
        
        out = F.relu(self.enc_conv1(x))
        out = F.relu(self.enc_conv2(out))
        out = F.relu(self.enc_conv3(out))
        out = F.relu(self.enc_conv4(out))
        out = out.view(batch_size,1024)
        
        mu = self.mu(out)
        logvar = self.logvar(out)
        
        return mu, logvar
        
    def decode(self, z):
        batch_size = z.shape[0]
        
        out = z.view(batch_size, self.latent_size, 1, 1)
        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = F.relu(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        
        return out
        
        
    def latent(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        
        return z
    
    def obs_to_z(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        
        return z

    def sample(self, z):
        out = self.decode(z)
        
        return out
    
    def vae_loss(self, out, y, mu, logvar):
        BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL, BCE, KL

    def get_latent_size(self):
        
        return self.latent_size

    def set_device(self, device):
        self.device = device

