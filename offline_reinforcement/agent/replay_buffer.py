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

    def __init__(self, buffer_path, history=4, suffixes = None, n_ckpts: int = 1) -> None:
        self.data = {}
        self.buffer_path = buffer_path
        self.n_ckpts = n_ckpts

        if suffixes:
            self.load_new_buffer(suffixes)
        else:
            self.load_new_buffer()

        self.history = history

    def get_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32).to(device)
        batch_next_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32).to(device)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long).to(device)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32).to(device)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int).to(device)
        data_indicies = np.random.choice(self.n_ckpts, size=batch_size)
        rand_indicies = np.random.choice(len(self.data['action'][0]) - (self.history + 1), size=batch_size, replace=False)
        for idx, rnd_idx, data_index in enumerate(zip(rand_indicies, data_indicies)):
            batch_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rnd_idx:rnd_idx+self.history, :, :])
            batch_next_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rnd_idx+1:rnd_idx+self.history+1, :, :])
            batch_actions[idx, :] = self.data['action'][data_index][rnd_idx + self.history]
            batch_reward[idx, :] = torch.from_numpy(np.asarray(self.data['reward'][data_index][rnd_idx+self.history]))
            batch_done[idx, :] = self.data['terminal'][data_index][rnd_idx+self.history]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    def load_new_buffer(self, suffixes: int = None):
        self.data = {}
        if not suffixes:
            suffixes = np.random.randint(low=0, high=50, size=self.n_ckpts)
        for elem in ELEMS:
            paths = [f'{self.buffer_path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz' for suffix in suffixes]
            files = (gzip.open(p, 'rb') for p in paths)
            self.data[elem] = [np.load(f) for f in files]
            [f.close() for f in files]


    def get_static_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32)
        batch_next_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int)
        data_index = np.random.choice(self.n_ckpts)
        for index, rand_index in enumerate(range(batch_size)):
            batch_state[index, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rand_index: rand_index + self.history, :, :])
            batch_next_state[index, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rand_index + 1: rand_index + self.history + 1, :, :])
            batch_actions[index, :] = self.data['action'][data_index][rand_index + self.history]
            batch_reward[index, :] = torch.from_numpy(np.asarray(self.data['reward'][data_index][rand_index + self.history]))
            batch_done[index, :] = self.data['terminal'][data_index][rand_index + self.history]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done
