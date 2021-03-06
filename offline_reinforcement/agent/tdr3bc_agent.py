import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.networks import Actor, Critic, CriticREM
from agent.replay_buffer_d4rl import ReplayBufferD4RL
import numpy as np
import matplotlib.pyplot as plt
import wandb
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3BC:
    def __init__(self, actor, actor_target, critic_1, critic_1_target,
                 critic_2, critic_2_target, actor_optimizer, critic_1_optimizer,
                 critic_2_optimizer, tau, dataset, batch_size, gamma, noise_std, noise_c,
                 min_action, max_action, alpha, action_dim, task):

        self.task = task

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
        self.action_dim = action_dim
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

            noise = torch.clamp(torch.empty(self.batch_size, self.action_dim).normal_(mean=0, std=self.noise_std), -self.noise_c, self.noise_c)
            next_actions = torch.clamp(self.actor_target(next_states) + noise,
                                       self.min_action.detach().numpy()[0], self.max_action.detach().numpy()[0])

            if isinstance(self.critic_1, CriticREM):

                # random weights
                alphas1 = np.random.uniform(low=0, high=1, size=self.critic_1.num_heads)
                alphas1 = alphas1/np.sum(alphas1)
                alphas2 = np.random.uniform(low=0, high=1, size=self.critic_1.num_heads)
                alphas2 = alphas2/np.sum(alphas2)

                Q_val_1 = self.critic_1_target(next_states, next_actions, alphas1)
                Q_val_2 = self.critic_2_target(next_states, next_actions, alphas2)

            else:

                Q_val_1 = self.critic_1_target(next_states, next_actions)
                Q_val_2 = self.critic_2_target(next_states, next_actions)

            td_targets = rewards + self.gamma * torch.minimum(Q_val_1, Q_val_2) * (1.0 - done)

        if isinstance(self.critic_1, CriticREM):

            Q_pred_1 = self.critic_1(states, actions, alphas1)
            Q_pred_2 = self.critic_2(states, actions, alphas2)

        else:

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
            if isinstance(self.critic_1, CriticREM):
                alphas = np.full(shape=self.critic_1.num_heads, fill_value=1/self.critic_1.num_heads)
                Q_pred = self.critic_1(states, pi, alphas)
            else:
                Q_pred = self.critic_1(states, pi)
            pre_loss = self.alpha/Q_pred.abs().mean().detach()
            actor_loss = - pre_loss * Q_pred.mean() + F.mse_loss(pi, actions)
            actor_loss.backward()
            self.actor_optim.step()

            # update the models
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

    def render_policy(self, time_step):

        if self.task == 'cheetah':
            state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
        elif self.task == 'humanoid':
            state = np.concatenate((time_step.observation['joint_angles'],
                                    np.expand_dims(np.array(time_step.observation['head_height']), axis=0),
                                    time_step.observation['extremities'], time_step.observation['torso_vertical'],
                                    time_step.observation['com_velocity'], time_step.observation['velocity']))

        state = (state-self.replay_buffer.mean)/self.replay_buffer.std
        state = torch.from_numpy(state).float()
        return self.act(state)

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

    def save(self, file_path, epoch):
        torch.save(self.critic_1.state_dict(), join(file_path, f"critic_1_epoch_{epoch}.pth"))
        torch.save(self.critic_2.state_dict(), join(file_path, f"critic_2_epoch_{epoch}.pth"))
        torch.save(self.actor.state_dict(), join(file_path, f"actor_epoch_{epoch}.pth"))

    def load(self, file_path, epoch):
        self.critic_1.load_state_dict(torch.load(os.path.join(file_path, f"critic_1_epoch_{epoch}.pth")))
        self.critic_2.load_state_dict(torch.load(os.path.join(file_path, f"critic_2_epoch_{epoch}.pth")))
        self.actor.load_state_dict(torch.load(os.path.join(file_path, f"actor_epoch_{epoch}.pth")))
