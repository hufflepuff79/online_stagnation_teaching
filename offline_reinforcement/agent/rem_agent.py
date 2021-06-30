import tensorflow as tf
import torch
import torch.nn as nn
from agent.networks import REM
from agent.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REMAgent:

    def __init__(self, Q: nn.Module, Q_target: nn.Module, num_actions: int, data_dir: str,
                 optimizer: torch.optim.Optimizer, batch_size: int = 32,
                 epsilon: int = 0.001, gamma: int = 0.99, history: int = 4):

        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.num_actions = num_actions

        # replay buffer
        self.replay_buffer = ReplayBuffer(data_dir, history=history)

        # state buffer
        self.state_buffer = StateBuffer(size=history)

        # parameters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        # optimizer
        self.optimizer = optimizer

        # loss
        self.loss_function = torch.nn.SmoothL1Loss(beta=1.0)

    def train_batch(self) -> None:

        # set network to train mode
        self.set_net_status(eval=False)

        # sample replay buffer
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = self.replay_buffer.get_minibatch(self.batch_size)

        # fig, axs = plt.subplots(2, 2)
        # axs = axs.flatten()
        # for i in range(len(axs)):
        #     axs[i].imshow(batch_states[0, i, :, :], cmap='gray')
        # plt.show()

        # random weights
        alphas = np.random.uniform(low=0, high=1, size=self.Q.num_heads)
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
        return loss

    def set_net_status(self, eval=True):
        """" Status of the networks set to train/eval"""
        if eval:
            self.Q.eval()
            self.Q_target.eval()
        else:
            self.Q.train()
            self.Q_target.train()

    def update_target(self) -> None:
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, state: torch.Tensor, deterministic: bool, distribution=None) -> int:

        self.state_buffer.update(state)
        r = np.random.uniform()

        # fig, axs = plt.subplots(2, 2)
        # axs = axs.flatten()
        # for i in range(len(axs)):
        #     axs[i].imshow(self.state_buffer.states[0, i, :, :], cmap='gray')
        # plt.show()

        alphas = np.full(shape=self.Q.num_heads, fill_value=1/self.Q.num_heads)

        if deterministic or r > self.epsilon:
            action_id = np.argmax(self.Q(self.state_buffer.get_states(), alphas).cpu().detach().numpy())
        else:
            action_id = np.random.choice(a=self.num_actions, p=distribution)

        return action_id


class StateBuffer:

    def __init__(self, size: int = 4, img_width: int = 84, img_height: int = 84):

        self.size = size
        self.img_width = img_width
        self.img_height = img_height
        self.states = torch.zeros(1, size, img_width, img_height, dtype=torch.float).to(device)

    def update(self, new_state):

        self.states = torch.roll(self.states, -1, 1).to(device)
        self.states[:, -1, :, :] = new_state.to(device)

    def reset(self, new_state):

        self.states = torch.zeros(1, self.size, self.img_width, self.img_height, dtype=torch.float).to(device)
        self.states[:, -1, :, :] = new_state

    def get_states(self):

        return self.states
