import tensorflow as tf
import torch
import torch.nn as nn
from agent.networks import REM
from agent.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REMAgent:

    def __init__(self, Q: nn.Module, Q_target: nn.Module, num_actions: int, data_dir: str,
                 optimizer: torch.optim.Optimizer, batch_size: int = 32,
                 epsilon: int = 0.001, gamma: int = 0.99, history: int = 4, suffixes = None, n_ckpts = 1):

        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.num_actions = num_actions

        # replay buffer
        self.replay_buffer = ReplayBuffer(data_dir, history=history, suffixes=suffixes, n_ckpts=n_ckpts)

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

        # history
        self.history = history

    def train_batch(self, logging: bool = False, epoch: int = 0) -> None:

        # set network to train mode
        self.set_net_status(eval=False)

        # sample replay buffer
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = self.replay_buffer.get_minibatch(self.batch_size)

        # random weights
        alphas = np.random.uniform(low=0, high=1, size=self.Q.num_heads)
        alphas = alphas/np.sum(alphas)

        # update
        with torch.no_grad():
            max_action_Qs, _ = torch.max(self.Q_target(batch_next_states, alphas), dim=1)
            max_action_Qs = torch.unsqueeze(max_action_Qs, 1)
            td_targets = batch_rewards + self.gamma * max_action_Qs * (1.0-batch_done)

        self.optimizer.zero_grad()
        
        Q_pred = self.Q(batch_states, alphas).gather(dim=1, index=batch_actions)
        loss = self.loss_function(Q_pred, td_targets)
        loss.backward()
        self.optimizer.step()

        log = {}

        if logging:
            log["states"] = [wandb.Image(batch_states[0, i, :, :], caption=f"state {i}") for i in range(self.history)]
            log["next_states"] = [wandb.Image(batch_next_states[0, i, :, :], caption=f"next state {i}") for i in range(self.history)]
            log["max_Q"] = torch.max(Q_pred)
            log["min_Q"] = torch.min(Q_pred)

        return loss.detach(), log

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

        """
        fig, axs = plt.subplots(1, self.history)
        axs = axs.flatten()
        for i in range(self.history):
             axs[i].imshow(self.state_buffer.states[0, i, :, :], cmap='gray')
        plt.show()
        """

        alphas = np.full(shape=self.Q.num_heads, fill_value=1/self.Q.num_heads)

        if deterministic or r > self.epsilon:
            action_id = np.argmax(self.Q(self.state_buffer.get_states(), alphas).cpu().detach().numpy())
        else:
            action_id = np.random.choice(a=self.num_actions, p=distribution)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)


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
