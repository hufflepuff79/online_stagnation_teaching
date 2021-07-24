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
# from dm_control import suite

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params, seed: int = 42, log_wb: bool = False, logging_freq: int = 1000, use_rem: bool = False):

    # Training seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    if params.env_type == 'dm_control':
        # Import correct libs
        from dm_control import suite, viewer
        # get the envirionment infos
        if params.env_name == 'cheetah':
            env = suite.load('cheetah', 'run')
            action_space = env.action_spec().shape[0]
            max_action = torch.from_numpy(env.action_spec().maximum.astype(np.float32)).to(device)
            min_action = torch.from_numpy(env.action_spec().minimum.astype(np.float32)).to(device)
            observation_space = 17
            env.close()
        elif params.env_name == 'humanoid':
            env = suite.load('humanoid', 'run')
            action_space = env.action_spec().shape[0]
            max_action = torch.from_numpy(env.action_spec().maximum.astype(np.float32)).to(device)
            min_action = torch.from_numpy(env.action_spec().minimum.astype(np.float32)).to(device)
            observation_space = 67
            env.close()
        else:
            print('Unsupported env_name for gym envirionment. Use cheetah or humanoid')
            return
        
        # Load the dataset, using_d4rl=using_d4rl
        try:
            path = ''
            path = join(params.data_dir, params.env_name + '_data.pkl')
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print(e)
            print(f'Unable to load the pickled data from {path}')
            return
    elif params.env_type == 'gym':
        import d4rl
        import gym
        if params.env_name == 'cheetah':
            env = gym.make('halfcheetah-expert-v2')
        else:
            print('Unsupported env_name for gym envirionment. Use cheetah')
            return
        dataset = d4rl.qlearning_dataset(env)
        action_space = env.action_space.shape[0]
        max_action = torch.from_numpy(env.action_space.high).to(device)
        min_action = torch.from_numpy(env.action_space.low).to(device)
        observation_space = env.observation_space.shape[0]
        env.close()
    else:
        print('Unsupported environment type. Use dm_control or gym.')
        return

    # create networks 
    actor = Actor(max_action=max_action, in_features=observation_space, out_features=action_space).to(device)
    actor_target = Actor(max_action=max_action, in_features=observation_space, out_features=action_space).to(device)

    if use_rem:
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
    sp.add_bar("valid", "Validation Progress", params.validation_runs * 1000)

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
        total_reward = online_validation(agent=agent, env_name=params.env_name, env_type=params.env_type, seed=seed, num_episodes=params.validation_runs, status_func=sp.increment_and_print,
                                        status_arg="valid")
        sp.done_element("valid")
      
        print(total_reward)
        if log_wb:
            wandb.log({'Average Reward' : total_reward, 'epoch': epoch})

        # save weights at regular intervals
        if epoch % params.agent_save_weights == 0:
            agent.save(log_dir, epoch)


def online_validation(agent, env_name, env_type, seed=42, num_episodes=10, status_func=lambda *args :None, status_arg=None, render=False):
    if env_type == 'dm_control':
        # Import correct libs
        from dm_control import suite, viewer
        # get the envirionment infos
        if env_name == 'cheetah':
            env = suite.load('cheetah', 'run', task_kwargs={'random' : seed})
        elif env_name == 'humanoid':
            env = suite.load('humanoid', 'run', task_kwargs={'random' : seed})
        else:
            print('Unsupported env_name for gym envirionment. Use cheetah or humanoid')
            return
    elif env_type == 'gym':
        import d4rl
        import gym
        if env_name == 'cheetah':
            env = gym.make('halfcheetah-expert-v2')
            env.seed(seed)
            env.action_space.seed(seed)
        else:
            print('Unsupported env_name for gym envirionment. Use cheetah')
            return
    else:
        print('Unsupported environment type. Use dm_control or gym.')
        return

    total_reward = 0
    for _ in range(num_episodes):
        done = False
        if env_type == 'gym':
            state = env.reset()
            state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
            state = torch.from_numpy(state).float()
        else:
            unplugged_rl_step_count = 0
            time_step = env.reset()
            if env_name == "cheetah":
                state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
            else:
                state = np.concatenate((time_step.observation['velocity'], time_step.observation['com_velocity'],
                                        time_step.observation['torso_vertical'], time_step.observation['extremities'],
                                        np.expand_dims(np.array(time_step.observation['head_height']), axis=0), time_step.observation['joint_angles']))

            state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
            state = torch.from_numpy(state).float()

        while not done:
            action = agent.act(state)
            if env_type == 'gym':
                state, reward, done, _ = env.step(action)
                state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
                state = torch.from_numpy(state).float()
                if render:
                    env.render()
                    time.sleep(0.01)
            else:
                time_step = env.step(action)
                if render:
                    viewer.launch(env, agent.render_policy)

                if env_name == "cheetah":
                    state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
                else:
                    state = np.concatenate((time_step.observation['velocity'], time_step.observation['com_velocity'],
                                        time_step.observation['torso_vertical'], time_step.observation['extremities'],
                                        np.expand_dims(np.array(time_step.observation['head_height']), axis=0), time_step.observation['joint_angles']))

                state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
                state = torch.from_numpy(state).float()
                reward = time_step.reward
                unplugged_rl_step_count += 1
                if unplugged_rl_step_count >= 1000:
                    done = True
            total_reward += reward
            status_func(status_arg)
    total_reward /= num_episodes
    if env_type == 'gym':
        total_reward = env.get_normalized_score(total_reward)

    env.close()
    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--env_name', type=str, default='cheetah', help='Name of the environment (cheetah or human)')
    parser.add_argument('--env_type', type=str, default='gym', help='Type of the environment (dm_control or gym)')
    parser.add_argument('--data_dir', type=str, help='Path to the transition data of the unplugged_rl dataset')
    parser.add_argument('--epochs', type=int, help='Number of test evaluations. Every epoch there are <iterations> iterations')
    parser.add_argument('--iterations', type=int, help='aNumber of iterations per epoch. Agent gets and trains one batch per epoch')
    parser.add_argument('--validation_runs', type=int, default=10, help='How many runs to make for test evaluation. Average over runs is returned')
    parser.add_argument("--agent_save_weights", type=int, default=100, help="Frequency at which the weights of network are saved")
    parser.add_argument("--wandb", action='store_true', help="Log with wandb")
    parser.add_argument("--wandb_name",type=str, help="set a fixed wandb run name", default=None)
    parser.add_argument('--use_rem', action='store_true', help='Use a rem like critic.')
    parser.add_argument('--seed', type=int, default=42, help='Use a rem like critic.')
    
    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/td3+bc_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])
    
    params.fix()

    train(params, log_wb=args.wandb, seed=args.seed, use_rem=args.use_rem)
