import os
from mpi4py import MPI
from Gym_Dataset import Gym_Dataset

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

env_name = "CarRacing-v0"
start_id = 2000
slave_rollouts = 250
save_path = os.path.dirname(os.path.abspath(__file__))

def master():
    
    if not os.path.isdir(save_path + "\\" + env_name + "_dataset"): 
        os.mkdir(save_path + "\\" + env_name + "_dataset")
    
    num_slaves = comm.Get_size()-1
    
    rollouts_index = range(start_id,start_id + slave_rollouts*num_slaves)
    for slave in range(1, num_slaves+1):

        packet = rollouts_index[(slave-1)*slave_rollouts:((slave-1)*slave_rollouts)+slave_rollouts]
        packet = ("rollouts", packet)
        comm.send(packet, dest=slave)
        
    for slave in range(1, num_slaves+1):
        packet = ("done", None)
        comm.send(packet, dest=slave)
    
        
def slave():
    
    msg, rollout_names = comm.recv(source=0)
    
    if msg == "done":
        return
    
    gd = Gym_Dataset(env_name)
    gd.get(rollout_names)
    

if __name__ == "__main__":

    if rank == 0:
        master()
    else:
        slave()
