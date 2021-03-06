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

    def __init__(self, buffer_path, history: int = 4, suffixes: list = None, n_ckpts: int = 1) -> None:
        self.data = {}
        self.buffer_path = buffer_path
        self.n_ckpts = n_ckpts

        self.history = history

    def get_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32).to(device)
        batch_next_state = torch.empty(batch_size, self.history, 84, 84, dtype=torch.float32).to(device)
        batch_actions = torch.empty(batch_size, 1, dtype=torch.long).to(device)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32).to(device)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int).to(device)
        data_indicies = np.random.choice(self.n_ckpts, size=batch_size)
        rand_indicies = np.random.choice(len(self.data['action'][0]) - (self.history + 1), size=batch_size, replace=False)
        for idx, (rnd_idx, data_index) in enumerate(zip(rand_indicies, data_indicies)):
            batch_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rnd_idx:rnd_idx+self.history, :, :])
            batch_next_state[idx, :, :, :] = torch.from_numpy(self.data['observation'][data_index][rnd_idx + 1:rnd_idx+self.history + 1, :, :])
            batch_actions[idx, :] = self.data['action'][data_index][rnd_idx + self.history - 1]
            batch_reward[idx, :] = torch.from_numpy(np.asarray(self.data['reward'][data_index][rnd_idx+self.history - 1]))
            batch_done[idx, :] = self.data['terminal'][data_index][rnd_idx + self.history - 1]

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    def load_new_buffer(self, suffixes: int = None, use_splits: bool = False, max_suffix: int = 50):
        self.data = {}
        if not suffixes:
            suffixes = np.random.randint(low=0, high=max_suffix, size=self.n_ckpts)
        for elem in ELEMS:
            if use_splits:
                paths = [f'{self.buffer_path}{STORE_FILENAME_PREFIX}{elem}_split_ckpt.{suffix}.gz' for suffix in suffixes]
            else:
                paths = [f'{self.buffer_path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz' for suffix in suffixes]
            files = (gzip.open(p, 'rb') for p in paths)
            self.data[elem] = [np.load(f) for f in files]
            [f.close() for f in files]
