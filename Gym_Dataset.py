import gym
import numpy as np
import pickle
import cv2
import os
import torch

from World_Model import World_Model
from Env_Runner import Env_Runner

class Gym_Dataset:
    
    def __init__(self, env_name, device=None):
        
        self.env_name = env_name

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")
            

    # using MPI, rollout_names -> filenames for the rollouts (numbers)
    
    def get(self, rollout_names, img_resize=(64,64), save_path=os.path.dirname(os.path.abspath(__file__))):
        
        env = gym.make(self.env_name)
        
        for i in rollout_names:

            #different random policy for every rollout
            policy = World_Model("random vae",
                                 "random mdn rnn",
                                  3, self.device, random=True)
            
            runner = Env_Runner(self.device)

            # let the agent start at random track tile to enrich vae and mdnrnn
            obs, actions, rewards = runner.run(env, policy, img_resize=(64,64), random_start=True)
            
            data = {"obs":np.array(obs),"actions":np.array(actions),"rewards":np.array(rewards)}
            file = open(save_path + "\\" + self.env_name + f'_dataset\\{i}',"wb")
            pickle.dump(data, file)
            file.close()
        
        env.close()
