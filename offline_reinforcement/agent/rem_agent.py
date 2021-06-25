import tensorflow as tf
import torch
import torch.nn as nn
from agent.networks import REM
from agent.replay_buffer import ReplayBuffer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REMAgent:

    def __init__(self, Q: nn.Module, Q_target: nn.Module, num_actions : int, data_dir: str):
        
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.num_actions = num_actions

        # replay buffer
        self.replay_buffer = ReplayBuffer(data_dir)

        # state buffer
        self.state_buffer = StateBuffer(size=4)

        # parameters
        self.batch_size = 32 
        self.epsilon = 0.001
        self.gamma = 0.99

        # optimizer
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.00005, eps=0.0003125)

        # loss
        self.loss_function = torch.nn.SmoothL1Loss(beta=1.0)

    def train_batch(self) -> None:
        
        # sample replay buffer
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = self.replay_buffer.get_minibatch(self.batch_size)
         
        # random weights
        alphas = np.random.uniform(low=0, high=1, size=200)
        alphas = alphas/np.sum(alphas)

        # update 
        with torch.no_grad():
            max_action_Qs, _ = torch.max(self.Q_target(batch_next_states, alphas), dim=1)
            td_targets = batch_rewards + self.gamma * max_action_Qs * (1.0-batch_done)

        self.optimizer.zero_grad()
        Q_pred = self.Q(batch_states, alphas)[torch.arange(self.batch_size), batch_actions]
        loss = self.loss_function(Q_pred, td_targets)
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, state: torch.Tensor, deterministic: bool, distribution=None) -> int:
        
        self.state_buffer.update(state)
        r = np.random.uniform()
        alphas = np.random.uniform(low=0, high=1, size=200)
        alphas = alphas/np.sum(alphas)

        if deterministic or r > self.epsilon:
            action_id = np.argmax(self.Q(self.state_buffer.get_states(), alphas).detach().numpy())
        else:
            action_id = np.random.choice(a=self.num_actions, p=distribution)

        return action_id


class StateBuffer:

    def __init__(self, size: int=4, img_width: int=84, img_height: int=84):

        self.size = size
        self.states = torch.zeros(1, size, img_width, img_height, dtype=torch.float)

    def update(self, new_state):

        self.states = torch.roll(self.states, -1, 1)
        self.states[:, -1, :, :] = new_state

    def reset(self, new_state):

        self.states = torch.cat((new_state,)*4, dim=1)

    def get_states(self):

        return self.states
