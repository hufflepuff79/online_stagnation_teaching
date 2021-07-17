import torch
import torch.nn as nn
from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3BC:
    def __init__(self, actor, actor_target, critic_1, critic_1_target,
                 critic_2, critic_2_target):
        self.actor = actor
        self.actor_target = actor_target

        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.critic_1_target = self.critic_1_target
        self.critic_2_target = self.critic_2_target

        # set same parameters for target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1_target.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2_target.state_dict())

        # replay buffer
        self.replay_buffer = ReplayBuffer(data_dir, history=history, suffixes=suffixes, n_ckpts=n_ckpts)

        # parameters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma

        # optimizer
        self.optimizer = optimizer

        # loss
        self.loss_function = torch.nn.SmoothL1Loss(beta=1.0)

    def train(self):

        states, actions, rewards, next_states, done = self.replay_buffer.get_minibatch(self.batch_size)

        # dont forget to add the noise to the

        with torch.no_grad():
            max_Q_1 = self.critic_1_target(next_states)
            max_Q_2 = self.critic_2_target(next_states)

            td_targets = rewards + self.gamma * torch.min(max_Q_1, max_Q_2) * (1.0 - done)

        Q_pred = self.critic_1(states)

        self.optimizer.zero_grad()
        loss = self.loss_function(Q_pred, td_targets)
        loss.backward()
        self.optimizer.step()

        # update the actor network

    def set_net_status(self, eval=True):
        """" Status of the networks set to train/eval"""
        if eval:
            self.actor.eval()
        else:
            self.actor.train()


    def update_target_actor(self):
        # update the target actor network here


    def update_target_critic(self):
        # update the target critic networks here
        self.critic_1_target.load_state_dict(self.critic_1_target.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2_target.state_dict())

    def act(self, state):
        action = self.actor(state)
        return action

    def save(self):
        torch.save(self.Q.state_dict(), file_name)
