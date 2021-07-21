import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.networks import Actor, Critic
from agent.replay_buffer_d4rl import ReplayBufferD4RL
import numpy as np
import matplotlib.pyplot as plt
import wandb
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3BC:
    def __init__(self, actor, actor_target, critic_1, critic_1_target,
                 critic_2, critic_2_target, actor_optimizer, critic_1_optimizer,
                 critic_2_optimizer, tau, dataset, batch_size, gamma, noise_std, noise_c
                 min_action, max_action, alpha):
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

        self.replay_buffer = ReplayBufferD4RL(dataset)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_c = noise_c
        self.min_action = min_action
        self.max_action = max_action
        self.alpha = alpha

        # optimizer
        self.critic_1_optim = critic_1_optimizer
        self.critic_2_optim = critic_2_optimizer
        self.actor_optim = actor_optimizer

        # loss
        self.critic_loss_func = nn.MSELoss()

    def train_batch(self, optim_actor=False):

        self.set_net_status(eval=False)

        states, actions, rewards, next_states, done = self.replay_buffer.get_minibatch(self.batch_size)

        with torch.no_grad():

            noise = torch.clamp(torch.empty(batch_size).normal_(mean=0,std=self.noise_std), -self.noice_c, self.noise_c)
            next_actions = torch.clamp(self.actor_target(next_states) + noise, self.min_action, self.max_action)

            Q_val_1 = self.critic_1_target(next_states, next_actions)
            Q_val_2 = self.critic_2_target(next_states, next_actions)

            # TODO: is using torch.minimum correct here? this will take the min between the two elements
            # TODO: for each element pair of the two tensors, and also look at done shape/vals
            td_targets = rewards + self.gamma * torch.minimum(Q_val_1, Q_val_2) * (1.0 - done)

        Q_pred_1 = self.critic_1(states, actions)
        Q_pred_2 = self.critic_2(states, actions)

        # optimize critic networks
        self.critic_1_optim.zero_grad()
        loss_c1 = self.critic_loss_func(Q_pred_1, td_targets)
        loss_c1.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        loss_c2 = self.critic_loss_func(Q_pred_2, td_targets)
        loss_c2.backward()
        self.critic_2_optim.step()

        if optim_actor:
            self.actor_optim.zero_grad()
            
            # loss function
            pi = self.actor(states)
            Q_pred = self.critic_1(states, pi)
            l = self.alpha/Q_pred.abs().mean().detach()
            actor_loss = - l * Q_pred.mean() + F.mse_loss(pi, actions)
            actor_loss.backward()
            self.actor_optim.step()

            #update the models
            self.update_target_actor()
            self.update_target_critic()

    def set_net_status(self, eval=True):
        """" Status of the networks set to train/eval"""
        if eval:
            self.actor.eval()
            self.critic_1.eval()
            self.critic_2.eval()
        else:
            self.actor.train()
            self.critic_1.train()
            self.critic_2.train()


    def update_target_critic(self):
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


    def update_target_actor(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def act(self, state):
        action = self.actor(state)
        return action.detach().numpy()

    def save(self, file_path):
        #TODO: careful that the path contains the epoch name, otherwise will overwrite every save step
        torch.save(self.critic_1.state_dict(), join(file_path, "critic_1.pth"))
        torch.save(self.critic_2.state_dict(), join(file_path, "critic_2.pth"))
        torch.save(self.actor.state_dict(), join(file_path, "actor.pth"))
