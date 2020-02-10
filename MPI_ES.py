import pickle
import os
import time
import gym
import cma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mpi4py import MPI
from World_Model import World_Model
from Env_Runner import Env_Runner

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

device = torch.device('cpu')

# hard coded
hidden_size = 256
actions = 3
latent_size = 32

dirname = os.path.dirname(os.path.abspath(__file__))
save_path_controller = dirname +"\\"+"es_controller"

# es algorithm
eps=250
pop_size=16
num_rollouts=18

def master():

    es = cma.CMAEvolutionStrategy((((hidden_size+latent_size)*actions)+actions)*[0], 0.125,{'popsize': pop_size})
    num_slaves = comm.Get_size()-1
    
    f = open("log.txt", "w")
    f.close()
    f = open("log_pop_performance.csv", "w")
    f.close()

    os.mkdir(save_path_controller)
    
    for ep in range(eps):
        
        solutions = es.ask()

        sol = []
        fitness = []

        slave_jobs = pop_size//num_slaves

        start_time = time.time()

        for slave in range(1, num_slaves+1):
            
            packet = solutions[(slave-1)*slave_jobs:((slave-1)*slave_jobs)+slave_jobs]
            packet = ("fitness", packet)
            comm.send(packet, dest=slave)

        for slave in range(1, num_slaves+1):

            slave_result = comm.recv(source=slave)
            
            for i in range(len(slave_result)):  
                sol.append(slave_result[i][0])
                fitness.append(slave_result[i][1])

        es.tell(sol,fitness)
        
        end_time = time.time()
        best_reward = es.result[1]
        curr_reward = es.result[2]

        fitness = - np.array(fitness)
        fit_max = np.amax(fitness)
        fit_mean = np.mean(fitness)
        fit_min = np.amin(fitness)
        f = open("log_pop_performance.csv", "a+")
        #min, avg, max
        f.write(f"{fit_min},{fit_mean},{fit_max}\n")
        f.close()

        f = open("log.txt", "a+")
        f.write("***********************\n")
        f.write(f"ep: {ep} finished | time: {end_time - start_time}\n")
        f.write(f"best reward: {best_reward}\n")
        f.write(f"best reward batch: {curr_reward}\n")
        f.close()

        weights = es.result[0]
        f = open(save_path_controller + f"\\network_{ep}.pt","wb")
        pickle.dump(weights,f)
        f.close()
        

    for slave in range(1, num_slaves+1):
        packet = ("done", None)
        comm.send(packet, dest=slave)
        
    
def slave():
    
    env = gym.make("CarRacing-v0")
    
    while True:
        
        msg, solutions = comm.recv(source=0)

        if msg == "done":
            return
        
        fitness = worker(solutions, env)

        packet = []
        for i in range(len(solutions)):
            packet.append([solutions[i],fitness[i]])
        
        comm.send(packet, dest=0)

def worker(solutions, env):

    
    fitness_solutions = []

    if not isinstance(solutions, list):
        solutions = [solutions]
    
    for weights in solutions:

        wm = World_Model(dirname + "\\vae.pt",
                     dirname + "\\mdn_rnn.pt",
                     actions,
                     device)
        
        w = weights[0:actions*(hidden_size+latent_size)]
        b = weights[actions*(hidden_size+latent_size)::]
        
        w = nn.Parameter(torch.tensor(np.reshape(w,(actions,hidden_size+latent_size))).type('torch.FloatTensor').to(device))
        b = nn.Parameter(torch.tensor(b).type('torch.FloatTensor').to(device))
        
        wm.set_controller(w,b)

        fitness = []
        for i in range(num_rollouts):
            
            runner = Env_Runner(device)
            wm.reset_rnn()
            _, _, rewards = runner.run(env, wm, img_resize=(64,64))

            # append negative return, because ES will try to minimize it
            fitness.append(-np.sum(np.array(rewards)))

        env.close()
        
        fitness_solutions.append(np.mean(np.array(fitness)))
    
    return fitness_solutions


if __name__ == "__main__":

    if rank == 0:
        master()
    else:
        slave()
