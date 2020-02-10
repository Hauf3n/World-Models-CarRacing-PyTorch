import gym
import numpy as np
import pickle
import cv2
import os

#from Env_Runner import Env_Runner

class Gym_Dataset:
    
    def __init__(self, env_name):
        
        self.env_name = env_name

    # using MPI, rollout_names -> filenames for the rollouts (numbers)    
    def get(self, rollout_names, img_resize=(64,64), save_path=os.path.dirname(os.path.abspath(__file__))):
        
        #os.mkdir(save_path + "\\" + self.env_name + "_dataset")
        env = gym.make(self.env_name)
        
        for i in rollout_names:
            
            obs = []
            actions = []
            rewards = []
            
            ob = env.reset()
            done = False
            action_steering = 0
            while not done:

                # let the car drive reasonably random rather than almost standing by using env.action.sample()
                action = env.action_space.sample()
                action[0] = 0.01 * np.random.randn(1) + action_steering
                action[1] = 0.05 * np.random.randn(1) + 0.01
                action[2] = 0
                
                if action[0] <= -0.2 or action[0] >= 0.2:
                    action_steering = 0
                else:
                    action_steering = action[0]
                
                actions.append(action)
                
                if img_resize is not None:
                    ob = ob[0:84,:,:]
                    ob = cv2.resize(ob, dsize=img_resize, interpolation=cv2.INTER_CUBIC)
                
                obs.append(ob)
                
                ob, r, done, _ = env.step(action)
                
                rewards.append(r)
            
            data = {"obs":np.array(obs),"actions":np.array(actions),"rewards":np.array(rewards)}
            file = open(save_path + "\\" + self.env_name + f'_dataset\\{i}',"wb")
            pickle.dump(data, file)
            file.close()
        
        env.close()
