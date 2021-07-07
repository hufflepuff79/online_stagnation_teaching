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
        rand_indicies = np.random.choice(len(self.data['action']) - (self.history + 1), size=batch_size, replace=False)
        for idx, rnd_idx in enumerate(rand_indicies):
            batch_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][rnd_idx:rnd_idx+self.history, :, :])
            batch_next_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][rnd_idx+1:rnd_idx+self.history+1, :, :])
            batch_actions[idx, :] = self.data['action'][rnd_idx + self.history]
            batch_reward[idx, :] = torch.from_numpy(np.asarray(self.data['reward'][rnd_idx+self.history]))
            batch_done[idx, :] = self.data['terminal'][rnd_idx+self.history]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    def load_new_buffer(self, suffixes: int = None):
        self.data = {}
        if not suffixes:
            suffixes = np.random.randint(low=0, high=50, size=self.n_ckpts)
        for elem in ELEMS:
            paths = [f'{self.buffer_path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz' for suffix in suffixes]
            files = (gzip.open(p, 'rb') for p in paths)
            data_in = [np.memmap(f) for f in files]
            self.data[elem] = np.vstack(data_in)


    def get_static_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32)
        batch_next_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int)
        for index, rand_index in enumerate(range(batch_size)):
            batch_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index: rand_index + self.history, :, :])
            batch_next_state[index, :, :, :] = torch.from_numpy(self.data['observation'][rand_index + 1: rand_index + self.history + 1, :, :])
            batch_actions[index, :] = self.data['action'][rand_index + self.history]
            batch_reward[index, :] = torch.from_numpy(np.asarray(self.data['reward'][rand_index + self.history]))
            batch_done[index, :] = self.data['terminal'][rand_index + self.history]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done
