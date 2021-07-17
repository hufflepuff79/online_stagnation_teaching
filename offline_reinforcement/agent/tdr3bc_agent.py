import torch
import torch.nn as nn
from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import wandb
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3BC:
    def __init__(self, actor, actor_target, critic_1, critic_1_target,
                 critic_2, critic_2_target, critic_1_optimizer, critic_2_optimizer,
                 actor_optimizer, tao):
        self.actor = actor
        self.actor_target = actor_target

        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.critic_1_target = critic_1_target
        self.critic_2_target = critic_2_target

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
        self.tao = tao

        # optimizer
        self.critic_1_optim = critic_1_optimizer
        self.critic_2_optim = critic_2_optimizer
        self.actor_optim = actor_optimizer

        # loss
        self.critic_loss_func = nn.MSELoss()
        # TODO : self.actor_loss_func =

    def train(self):

        self.set_net_status(eval=False)
        states, actions, rewards, next_states, done = self.replay_buffer.get_minibatch(self.batch_size)

        # dont forget to add the noise to the action

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_1 = self.critic_1_target(next_states, next_actions)
            Q_2 = self.critic_2_target(next_states, next_actions)

            # TODO: is using torch.minimum correct here? this will take the min between the two elements
            # for each element pair of the two tensors
            td_targets = rewards + self.gamma * torch.minimum(Q_1, Q_2) * (1.0 - done)

        Q_pred = self.critic_1(states, actions)

        # optimize critic networks
        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()
        #TODO
        # loss_1 =
        # loss_2 = self.critic_loss_func(Q_pred, )

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


    def update_target_critic(self):
        for layer, _ in self.critic_1:
            # apply a soft update to both critic target networks
            self.critic_1_target.state_dict()[layer] = self.tao * self.critic_1_target.state_dict()[layer] + \
                                                       (1-self.tao) * self.critic_1.state_dict()[layer]

            self.critic_2_target.state_dict()[layer] = self.tao * self.critic_2_target.state_dict()[layer] + \
                                                       (1 - self.tao) * self.critic_2.state_dict()[layer]


    def update_target_critic(self):
        # TODO!!

    def act(self, state):
        action = self.actor(state)
        return action

    def save(self, file_path):
        #TODO: careful that the path contains the epoch name, otherwise will overwrite every save step
        torch.save(self.critic_1.state_dict(), join(file_path, "critic_1.pth"))
        torch.save(self.critic_2.state_dict(), join(file_path, "critic_2.pth"))
        torch.save(self.actor.state_dict(), join(file_path, "actor.pth"))