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
import imageio

from VAE import VAE
from VAE import VAE_DataSet
from VAE_Trainer import VAE_Trainer

from MDN_RNN import MDN_RNN
from MDN_RNN import MDN_RNN_DataSet
from MDN_RNN_Trainer import MDN_RNN_Trainer

from Controller import Controller

from gym.envs.box2d.car_dynamics import Car

class Env_Runner():
    
    def __init__(self, device):
        super().__init__()

        self.device = device
        
    def run(self, env, model, img_resize=None, random_start=False):
        
        obs = []
        actions = []
        rewards = []
        
        ob = env.reset()

        if random_start: #CarRacing random track tile start
            position = np.random.randint(len(env.track))
            env.env.car = Car(env.env.world, *env.env.track[position][1:4])
        
        done = False
        while not done:

            if img_resize:
                ob = ob[0:84,:,:]
                ob = cv2.resize(ob, dsize=img_resize, interpolation=cv2.INTER_CUBIC)
                ob_model = torch.tensor(ob/255).view(1,img_resize[0],img_resize[1],3).permute(0,3,1,2).type('torch.FloatTensor')

            action = model(ob_model.to(self.device)).detach().cpu().numpy()[0]
            
            obs.append(ob)
            actions.append(action)
            
            ob, r, done, _ = env.step(action)
            
            rewards.append(r)
       
        return obs, actions, rewards

    def render(self, env, model, img_resize=(64,64), dream=False, random_start=False, video=False):
        
        ob = env.reset()

        if random_start:
            position = np.random.randint(len(env.track))
            env.env.car = Car(env.env.world, *env.env.track[position][1:4])
        
        done = False

        # save videos
        obs = []
        obs_reconstruction = []
        
        while not done:

            action = None
            
            if dream:

                action = model.forward_dream(ob.to(self.device)).detach().cpu().numpy()
                
            else:
            
                if img_resize:
                    ob = ob[0:84,:,:]
                    ob = cv2.resize(ob, dsize=img_resize, interpolation=cv2.INTER_CUBIC)
                    obs.append(ob) # video
                    
                    ob = torch.tensor(ob/255).view(1,img_resize[0],img_resize[1],3).permute(0,3,1,2).type('torch.FloatTensor')

                action = model(ob.to(self.device)).detach().cpu().numpy()[0]

                obs_reconstruction.append(model.vae(ob.to(self.device))[0][0].detach().cpu().permute(1,2,0).numpy()) # video
            
            env.render()
            ob, r, done, _ = env.step(action)

        if video:

            vid = []
            for x,y in zip(obs, obs_reconstruction):
                frame = np.zeros((64,128,3))
                frame[:,0:64,:] += x
                frame[:,64::,:] += y*255
                vid.append(frame)

            
            w = imageio.get_writer('vae_video.mp4', format='FFMPEG', mode='I', fps=30, quality=9)
            for img in vid:
                w.append_data(img.astype(np.uint8))
            w.close()
            
                
            
            
