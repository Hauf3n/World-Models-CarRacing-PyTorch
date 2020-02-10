import pickle
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import cv2
import cma

from VAE import VAE
from VAE import VAE_DataSet
from VAE_Trainer import VAE_Trainer

from MDN_RNN import MDN_RNN
from MDN_RNN import MDN_RNN_DataSet
from MDN_RNN_Trainer import MDN_RNN_Trainer

from Controller import Controller


class World_Model(nn.Module):
    
    def __init__(self, vae_path, mdn_rnn_path, output_size, device, random=False):
        super().__init__()

        self.device = device

        if random:
            self.vae = VAE(32).to(device)
            self.vae.set_device(self.device)
            self.mdn_rnn = MDN_RNN(35, 32, 512, 256, 5).to(device)
        else:
            self.vae = torch.load(vae_path, map_location=self.device)
            self.vae.set_device(self.device)
            self.mdn_rnn = torch.load(mdn_rnn_path, map_location=self.device)
        
        self.state = (torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device),
                      torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device))
        #self.action = torch.zeros((1,output_size)).to(self.device)
        self.action = None
        
        self.controller_input = self.mdn_rnn.get_hidden_size() + self.vae.get_latent_size()
        self.output_size = output_size
        
        self.controller = Controller(self.controller_input, self.output_size).to(device)
        
    def forward(self, x):
        
        mu, logvar = self.vae.encode(x)
        z = self.vae.latent(mu, logvar)

        controller_in = torch.cat((z, self.state[0][0].detach()),dim=1)
        self.action = self.controller(controller_in)

        mdn_rnn_in = torch.cat((z, self.action), dim=1)
        _, self.state = self.mdn_rnn.forward_lstm(mdn_rnn_in, self.state)

        # discard computational graph, only want h,c activations
        self.state = (self.state[0].detach(), self.state[1].detach())

        return self.action

    def forward_dream(self, z):

        if self.state == None:
            self.state = (torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device),
                      torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device))
        
        controller_in = torch.cat((z, self.state[0][0].detach()),dim=1)
        self.action = self.controller(controller_in)

        mdn_rnn_in = torch.cat((z, self.action), dim=1)
        _, self.state = self.mdn_rnn.forward_lstm(mdn_rnn_in, self.state)

        # discard computational graph, only want h,c activations
        self.state = (self.state[0].detach(), self.state[1].detach())

        return self.action

    def forward_dream_env(self, mdn_rnn_in):

        #mdn_rnn_in = torch.cat((z, action), dim=1)
        pi, sigma, mu, self.state = self.mdn_rnn(mdn_rnn_in, self.state)

        # discard computational graph, only want h,c activations
        self.state = (self.state[0].detach(), self.state[1].detach())
        #self.state = None
        return pi, sigma, mu
    
    def reset_rnn(self):
        self.state = (torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device),
                      torch.zeros((1,1,self.mdn_rnn.get_hidden_size())).to(self.device))
        #self.state = None
    
    def set_controller(self, weights, bias):
        self.controller.set_weights(weights)
        self.controller.set_bias(bias)
        self.controller.to(self.device)

