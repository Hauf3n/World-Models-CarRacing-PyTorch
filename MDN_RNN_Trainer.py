import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from VAE import VAE
from MDN import MDN
from MDN_RNN import MDN_RNN
from MDN_RNN import MDN_RNN_DataSet


class MDN_RNN_Trainer():
    
    def __init__(self, data_dir, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        
        self.dataset = MDN_RNN_DataSet(data_dir)
        self.dl = DataLoader(self.dataset, batch_size=96, shuffle=True)
        
        self.model = MDN_RNN(input_size, output_size, mdn_units, hidden_size, num_mixs).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
    def train(self, epochs):
        
        training_loss = []
        self.model.train(True)
        
        for epoch in range(epochs):
            print(f"**** EPOCH {epoch} ****")
            
            start_time = time.time()
            
            for i, eps_data in enumerate(self.dl):
                
                x = eps_data[:,0:-1,:]
                
                # extract only z out of data (actions are in it aswell)
                y = eps_data[:,1::,0:self.output_size]

                pi, sigma, mu, _ = self.model(x.cuda())
                loss = self.model.loss( y.cuda(), pi, mu, sigma)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                training_loss.append(loss.cpu().detach().numpy())
                print(epoch, loss.data)
                    
            end_time = time.time()
            print(f'time elapsed: {end_time - start_time} \n******************')
            
        return training_loss
    
    def predict(self, z, vae): # predict next latent vector given a seq >= 1
        
        pi, sigma, mu, _ = self.model(z.cuda())

        # sample
        z = np.random.gumbel(loc=0, scale=1, size=pi[:,-1,:,:].shape)
        k = (np.log(pi[:,-1,:,:].detach().cpu().numpy()) + z).argmax(axis=1)
        indices = (0, k, range(32))
        rn = torch.randn(1).cuda()
        sample = rn * sigma[:,-1,:,:][indices] + mu[:,-1,:,:][indices]

        out = vae.decode(sample.cuda()).permute(0,2,3,1).cpu().detach().numpy()

        mixtures = [torch.normal(mu, sigma)[0, -1, i, :] for i in range(5)]
        mixtures = torch.stack(mixtures)
        out_mix = vae.decode(mixtures.cuda()).permute(0,2,3,1).cpu().detach().numpy()
        
        _, axs = plt.subplots(1, 6, figsize=(16, 16))
        axs = axs.flatten()
        
        axs[0].imshow(out[0])
        axs[1].imshow(out_mix[0])
        axs[2].imshow(out_mix[1])
        axs[3].imshow(out_mix[2])
        axs[4].imshow(out_mix[3])
        axs[5].imshow(out_mix[4])
        plt.show()
        
    def get_model(self):
        return self.model

    def save_model(self, path):
        torch.save(self.model, path)

