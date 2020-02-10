import pickle
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from MDN import MDN
import gym
import cv2
from gym.spaces.box import Box
from gym.utils import seeding

from scipy.misc import imresize as resize
from scipy.misc import toimage as toimage
SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

class CarRacingDream(gym.Env):

  def __init__(self, world_model, device):

    self.wm = world_model
    self.vae = world_model.vae
    self.mdn_rnn = world_model.mdn_rnn
    self.device = device
    
    self.z_size = self.vae.get_latent_size()
    self.viewer = None
    self.frame_count = None
    self.z = None
    self.temperature = None # to do
    self.vae_frame = None
    
    self.reset()

  def reset(self, start_s=3):
    
    self.frame_count = 0
    self.wm.reset_rnn()
    
    f = open("D:\\Implementation_Schmidhuber" +"\\" +"CarRacing-v0_VAE_Z_dataset" + f'\\111', "rb")
    data = pickle.load(f)
    
    mu, logvar = data["mu"][start_s].to(self.device), data["logvar"][start_s].to(self.device)
    
    eps = torch.randn_like(logvar).to(self.device)
    z = (mu + eps * torch.exp(0.5*logvar)).to(self.device).unsqueeze(0)
    #z = mu.to(self.device).unsqueeze(0)
    
    self.z = z
    return self.z

  def next_state(self, action):
    
    rnn_in = torch.cat((self.z,action), dim=1).unsqueeze(0).to(self.device)
    pi, sigma, mu = self.wm.forward_dream_env(rnn_in)
    
    z = np.random.gumbel(loc=0, scale=1, size=pi[:,-1,:,:].shape)
    k = (np.log(pi[:,-1,:,:].detach().cpu().numpy()) + z).argmax(axis=1)
    #k = pi[:,-1,:,:].detach().cpu().numpy().argmax(axis=1)
    indices = (0, k, range(self.z_size))

    rn = torch.randn_like(sigma[:,-1,:,:][indices]).to(self.device)
    #state = rn * sigma[:,-1,:,:][indices]*0.25 + mu[:,-1,:,:][indices]
    state = mu[:,-1,:,:][indices]
    
    return state

  def step(self, action):
    self.frame_count += 1

    action = torch.tensor(action)
    next_z = self.next_state(action)
    
    reward = 0
    
    done = False
    if self.frame_count > 1200:
      done = True
    self.z = next_z
    
    return next_z, reward, done, {}

  def decode_obs(self, z):
    
    # decode the latent vector
    img = self.vae.decode(z.reshape(1, self.z_size))
    img = img.detach().cpu()[0]
    img = img.permute(1,2,0).numpy()
    return img

  def render(self, mode='human', close=False):

    img = self.decode_obs(self.z)
    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))

    if self.frame_count > 0:
      pass
      #toimage(img, cmin=0, cmax=255).save('output/'+str(self.frame_count)+'.png')

    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if mode == 'rgb_array':
      return img

    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)

    time.sleep(0.016)

  def close(self):
    self.viewer.close()
