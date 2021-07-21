import d4rl
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBufferD4RL():

    def __init__(self, dataset) -> None:
        self.data = dataset
        self.observation_dim = self.data['observations'].shape[1]
        self.action_dim = self.data['actions'].shape[1]


    def get_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.observation_dim, dtype=torch.float32).to(device)
        batch_next_state = torch.empty(batch_size, self.observation_dim, dtype=torch.float32).to(device)
        batch_actions = torch.empty(batch_size, self.action_dim, dtype=torch.float32).to(device)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32).to(device)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int).to(device)
        rand_indicies = np.random.choice(self.data['actions'].shape[0], size=batch_size, replace=False)
        for idx, rnd_idx in enumerate(rand_indicies):
            batch_state[idx, :] = torch.from_numpy(self.data['observations'][rnd_idx])
            batch_next_state[idx, :] = torch.from_numpy(self.data['next_observations'][rnd_idx])
            batch_actions[idx, :] = torch.from_numpy(self.data['actions'][rnd_idx])
            batch_reward[idx, :] = torch.from_numpy(np.asarray(self.data['rewards'][rnd_idx]))
            batch_done[idx, :] = torch.from_numpy(np.asarray(self.data['terminals'][rnd_idx]))

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done
        
        
    def get_static_minibatch(self, batch_size: int = 32):
        batch_state = torch.empty(batch_size, self.observation_dim, dtype=torch.float32).to(device)
        batch_next_state = torch.empty(batch_size, self.observation_dim, dtype=torch.float32).to(device)
        batch_actions = torch.empty(batch_size, self.action_dim, dtype=torch.float32).to(device)
        batch_reward = torch.empty(batch_size, 1, dtype=torch.float32).to(device)
        batch_done = torch.empty(batch_size, 1, dtype=torch.int).to(device)
        for idx, rnd_idx in enumerate(range(batch_size)):
            batch_state[idx, :] = torch.from_numpy(self.data['observations'][rnd_idx])
            batch_next_state[idx, :] = torch.from_numpy(self.data['next_observations'][rnd_idx])
            batch_actions[idx, :] = torch.from_numpy(self.data['actions'][rnd_idx])
            batch_reward[idx, :] = torch.from_numpy(np.asarray(self.data['rewards'][rnd_idx]))
            batch_done[idx, :] = torch.from_numpy(np.asarray(self.data['terminals'][rnd_idx]))

        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done