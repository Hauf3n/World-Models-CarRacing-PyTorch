import pickle
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from VAE import VAE
from VAE import VAE_DataSet

class VAE_Trainer():
    
    def __init__(self, data_dir, latent_size=32):
        self.data_dir = data_dir
        self.latent_size = latent_size
        
        self.dataset = VAE_DataSet(data_dir)
        self.dl = DataLoader(self.dataset, batch_size=1, shuffle=True) # sample whole eps
        
        self.model = VAE(latent_size).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, epochs, batch_size=128):
        
        training_loss = []
        bce_loss = []
        kl_loss = []
        
        for epoch in range(epochs):
            print(f"**** EPOCH {epoch} ****")
            
            start_time = time.time()
            self.model.train(True)
            
            for i, eps_data in enumerate(self.dl):
                train_dl = DataLoader(torch.flatten(eps_data["obs"],start_dim=0, end_dim=1), batch_size=batch_size, shuffle=True)
                
                for i, y in enumerate(train_dl):

                    self.optimizer.zero_grad()
                    
                    y = y.cuda()
                    out, mu, logvar = self.model(y)

                    loss, bce, kl = self.model.vae_loss(out, y, mu, logvar)
                    loss.backward()
                    self.optimizer.step()

                    training_loss.append(loss.cpu().detach().numpy())
                    bce_loss.append(bce.cpu().detach().numpy())
                    kl_loss.append(kl.cpu().detach().numpy())
                    
            end_time = time.time()
            print(f'time elapsed: {end_time - start_time} \n******************')
            
        return training_loss, bce_loss, kl_loss
        
    def get_model(self):
        return self.model
    
    def make_z_dataset(self, save_path):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        self.model.train(False)
        for i, eps_data in enumerate(self.dl):

            file = open( save_path + f'\\{i}' ,"wb")

            x = torch.tensor(eps_data["obs"][0]).cuda()        
            mu, logvar = self.model.encode(x)
            
            eps_data["mu"] = mu
            eps_data["logvar"] = logvar
            eps_data["obs"] = None

            pickle.dump(eps_data, file)
            file.close()

    def visualize_sample(self, z):

        num = z.shape[0]
        out = self.model.sample(z).permute(0,2,3,1).cpu().detach().numpy()

        _, axs = plt.subplots((num//10)+1, 10, figsize=(64, 64))
        axs = axs.flatten()
        for img, ax in zip(out, axs):
            ax.imshow(img)
        plt.show()
        
    
    def plot_x_hat(self, ep_num, img_num):
        
        f = open(self.data_dir+f'\\{ep_num}', "rb")
        data = pickle.load(f)
        data["obs"] = data["obs"]/255
        f.close()
        
        _, axs = plt.subplots(1, 2, figsize=(64, 64))
        img = data["obs"][img_num] 
        self.model.train(False)
        img_in = torch.tensor([data["obs"][img_num]],).type('torch.FloatTensor').permute(0,3,1,2).cuda()
        img_out, _, _ = self.model(img_in)
        img_out = img_out.permute(0,2,3,1).cpu().detach().numpy()[0]
        axs[0].imshow(img)
        axs[1].imshow(img_out)
        plt.show()

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path, device):
        self.model = torch.load(path, map_location=device)
        self.device = device
