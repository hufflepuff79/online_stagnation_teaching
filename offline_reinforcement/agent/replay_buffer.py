from os import replace
import os
import subprocess
import torch
import numpy as np
import gzip
import matplotlib.pyplot as plt

STORE_FILENAME_PREFIX = '$store$_'

ELEMS = ['observation', 'action', 'reward', 'terminal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():

    def __init__(self, buffer_path, suffix=None) -> None:
        self.data = {}
        self.buffer_path = buffer_path
        self.load_new_buffer(suffix)

        
    def get_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, 4, 84, 84, dtype=torch.float32).to(device)
        batch_next_state = torch.empty(batch_size, 4, 84, 84, dtype=torch.float32).to(device)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long).to(device)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32).to(device)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int).to(device)
        rand_indeces = np.random.choice(len(self.data['action']) - 5, size=batch_size, replace=False)
        for index, rand_index in enumerate(rand_indeces):
            batch_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index: rand_index+4, :, :])
            batch_next_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index+1: rand_index+5, :, :])
            batch_actions[index, :] = self.data['action'][rand_index+4]
            batch_reward[index, :] = torch.from_numpy(np.asarray(self.data['reward'][rand_index+4]))
            batch_done[index, :] = self.data['terminal'][rand_index+4]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    def load_new_buffer(self, suffix: int = None):
        del self.data
        self.data = {}
        if suffix == None:
            suffix = np.random.randint(low=0, high=50)
        for elem in ELEMS:
            filename = f'{self.buffer_path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz'
            with gzip.open(filename, 'rb') as infile:
                    self.data[elem] = np.load(infile) 

    def get_static_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, 4, 84, 84, dtype=torch.float32)
        batch_next_state = torch.empty(batch_size, 4, 84, 84, dtype=torch.float32)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int)
        for index, rand_index in enumerate(range(batch_size)):
            batch_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index: rand_index+4, :, :])
            batch_next_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index+1: rand_index+5, :, :])
            batch_actions[index, :] = self.data['action'][rand_index+4]
            batch_reward[index, :] = torch.from_numpy(np.asarray(self.data['reward'][rand_index+4]))
            batch_done[index, :] = self.data['terminal'][rand_index+4]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done
