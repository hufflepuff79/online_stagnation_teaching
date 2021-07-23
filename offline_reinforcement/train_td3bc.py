import pickle
import argparse
from matplotlib.pyplot import step
import torch
from utils import StatusPrinter, Parameters
from torch.utils.tensorboard import SummaryWriter
from agent.tdr3bc_agent import TD3BC
from agent.networks import Actor, Critic, CriticREM
import torch.nn as nn
import numpy as np
from os.path import exists, join
from os import makedirs
import wandb
import d4rl
import gym
# from dm_control import suite

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params, log_wb: bool = False, logging_freq: int = 1000):

    # Logging
    # create wandb for logging stats
    if log_wb:
        wandb.init(entity="online_stagnation_teaching", config=params.as_dict())
        if params.wandb_name:
            wandb.run.name = params.wandb_name

    # directory to log agent parameters
    log_dir = join("train_stats", wandb.run.name if log_wb else str(int(time.time())))
    if not exists(log_dir):
        makedirs(log_dir)

    # create the environment

    # get the offline dataset if data-dir is given we use the pickle and unplugged rl
    # else its d4rl
    using_d4rl = False
    try:
        with open(params.data_dir, 'rb') as f:
            dataset = pickle.load(f)
    except:
        using_d4rl = True
        env = gym.make(params.env_name)
        dataset = d4rl.qlearning_dataset(env)


    # get environment info
    # TODO: More flexibel and probably split up the training in two files one for d4rl and one for unplugged
    if using_d4rl:
        action_space = env.action_space.shape[0]
        max_action = torch.from_numpy(env.action_space.high).to(device)
        min_action = torch.from_numpy(env.action_space.low).to(device)
        observation_space = env.observation_space.shape[0]
        env.close()
    else:
        env = suite.load('cheetah', 'run')
        action_space = env.action_spec().shape[0]
        max_action = torch.from_numpy(env.action_spec().maximum.astype(np.float32)).to(device)
        min_action = torch.from_numpy(env.action_spec().minimum.astype(np.float32)).to(device)
        observation_space = 17
        env.close()

    # create networks 
    actor = Actor(max_action=max_action, in_features=observation_space, out_features=action_space).to(device)
    actor_target = Actor(max_action=max_action, in_features=observation_space, out_features=action_space).to(device)

    if params.use_rem:
        print("USING REM")
        critic_1 = CriticREM(in_features=observation_space+action_space).to(device)
        critic_2 = CriticREM(in_features=observation_space+action_space).to(device)
        critic_1_target = CriticREM(in_features=observation_space+action_space).to(device)
        critic_2_target = CriticREM(in_features=observation_space+action_space).to(device)
    else:
        critic_1 = Critic(in_features=observation_space+action_space).to(device)
        critic_2 = Critic(in_features=observation_space+action_space).to(device)
        critic_1_target = Critic(in_features=observation_space+action_space).to(device)
        critic_2_target = Critic(in_features=observation_space+action_space).to(device)

    # create optimizers 
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=params.actor_lr)
    critic_1_optimizer = optimizer = torch.optim.Adam(critic_1.parameters(), lr=params.critic_lr)
    critic_2_optimizer = optimizer = torch.optim.Adam(critic_2.parameters(), lr=params.critic_lr)


    # create the TD3+BC agent
    agent = TD3BC(actor, actor_target, critic_1, critic_1_target,critic_2, critic_2_target,
                   actor_optimizer, critic_1_optimizer, critic_2_optimizer, tau=params.tau,
                   dataset=dataset, batch_size=params.mini_batch_size, gamma=params.discount_factor,
                   noise_std=params.policy_noise, noise_c=params.policy_noise_clipping,
                   min_action=min_action, max_action=max_action, alpha=params.tdc_bc_alpha, action_dim=action_space)

    # for logging
    sp = StatusPrinter()
    sp.add_counter("epoch", "Epochs", params.epochs, 0, bold=True)
    sp.add_bar("iter", "Iteration Progress", params.iterations)
    sp.add_bar("valid", "Validation Progress", params.validation_runs * params.max_episode_steps)

    # initiate training
    print(f"\nStarting Training\nEpochs: {params.epochs}\nIterations per Epoch: {params.iterations}\n\n")
    for epoch in range(params.epochs):
        sp.increment_and_print("epoch")
        sp.print_statement("iter")
        sp.reset_element("iter")

        for iteration in range(params.iterations):
            agent.train_batch(optim_actor=iteration % params.policy_update_frequency)
            sp.increment_and_print("iter")

        sp.done_element("iter")

        sp.print_statement("valid")
        sp.reset_element("valid")
        total_reward = online_validation(agent=agent, env_name=params.env_name, num_episodes=params.validation_runs, status_func=sp.increment_and_print,
                                        status_arg="valid", using_d4rl=using_d4rl)
        sp.done_element("valid")
      
        print(total_reward)
        if log_wb:
            wandb.log({'Average Reward' : total_reward, 'epoch': epoch})

        # save weights at regular intervals
        if epoch % params.agent_save_weights == 0:
            agent.save(log_dir, epoch)


def online_validation(agent, env_name, num_episodes=10, status_func=lambda *args :None, status_arg=None, render=False, using_d4rl=True):
    if using_d4rl:
        env = gym.make(env_name)
    else:
        env = suite.load('cheetah', 'run')

    total_reward = 0
    for _ in range(num_episodes):
        done = False
        if using_d4rl:
            state = env.reset()
            state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
            state = torch.from_numpy(state).float()
        else:
            unplugged_rl_step_count = 0
            time_step = env.reset()
            state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
            state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
            state = torch.from_numpy(state).float()

        while not done:
            action = agent.act(state)
            if using_d4rl:
                state, reward, done, _ = env.step(action)
                state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
                state = torch.from_numpy(state).float()
                if render:
                    env.render()
                    time.sleep(0.01)
            else:
                time_step = env.step(action)
                state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
                state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
                state = torch.from_numpy(state).float()
                reward = time_step.reward
                unplugged_rl_step_count += 1
                if unplugged_rl_step_count >= 1000:
                    done = True
            total_reward += reward
            status_func(status_arg)
    total_reward /= num_episodes
    if using_d4rl:
        total_reward = env.get_normalized_score(total_reward)

    env.close()
    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--env_name', type=str, help='Name of the environment')
    parser.add_argument('--data_dir', type=str, help='Path to the transition data')
    parser.add_argument('--epochs', type=int, help='Number of test evaluations. Every epoch there are <iterations> iterations')
    parser.add_argument('--iterations', type=int, help='aNumber of iterations per epoch. Agent gets and trains one batch per epoch')
    parser.add_argument('--validation_runs', type=int, help='How many runs to make for test evaluation. Average over runs is returned')
    parser.add_argument("--optimizer", default="adam", help='Optimizer')
    parser.add_argument('--adam_learning_rate', type=float, help='Learning rate of ADAM optimizer')
    parser.add_argument("--agent_save_weights", type=int, help="Frequency at which the weights of network are saved")
    parser.add_argument("--wandb", action='store_true', help="Log with wandb")
    parser.add_argument("--wandb_name",type=str, help="set a fixed wandb run name", default=None)
    parser.add_argument("--critic_lr", type=float, default=3e-4, help='Learning rate of the critic')
    parser.add_argument("--actor_lr", type=float, default=3e-4, help='Learning rate of the actor')
    parser.add_argument("--mini_batch_size", default=256, help="Mini batch size")
    parser.add_argument("--discount_factor", default=0.99, help="Discount factor")
    parser.add_argument("--target_update_rate", default=5e-3, help="Target update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise term applied to the policy")
    parser.add_argument("--policy_noise_clipping", default=0.5, help="Noise clipping range")
    parser.add_argument("--policy_update_frequency", default=2, help="Frequency for the policy update")
    parser.add_argument("--tdc_bc_alpha", default=2.5, help="hyperparameter alpha")
    parser.add_argument('--max_episode_steps', type=int, help='Maximum steps per episode of agent.')
    parser.add_argument('--use_rem', type=bool, help='Use a rem like critic.')
    
    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/td3+bc_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])
    
    params.fix()

    train(params, log_wb=args.wandb)