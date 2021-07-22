import argparse
from matplotlib.pyplot import step
import torch
from utils import StatusPrinter, Parameters
from agent.tdr3bc_agent import TD3BC
from agent.networks import Actor, Critic
import torch.nn as nn
import numpy as np
from dm_control import suite
import time

if args.env == 'dm_control':
    import d4rl
elif args.env == 'gym':
    import gym



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(params, path, epoch, environment):


    # get environment info
    # TODO: More flexibel and probably split up the training in two files one for d4rl and one for unplugged
    if environment == 'dm_control':
        env = gym.make(params.env_name)
        action_space = env.action_space.shape[0]
        max_action = torch.from_numpy(env.action_space.high).to(device)
        min_action = torch.from_numpy(env.action_space.low).to(device)
        observation_space = env.observation_space.shape[0]
        env.close()
    
    elif environment == 'gym':
        env = suite.load('cheetah', 'run')
        action_space = env.action_spec().shape[0]
        max_action = torch.from_numpy(env.action_spec().maximum.astype(np.float32)).to(device)
        min_action = torch.from_numpy(env.action_spec().minimum.astype(np.float32)).to(device)
        observation_space = 17
        env.close()

    # create networks 
    actor = Actor(max_action=max_action, in_features=observation_space, out_features=action_space)
    actor_target = Actor(max_action=max_action, in_features=observation_space, out_features=action_space)

    critic_1 = Critic(in_features=observation_space+action_space)
    critic_2 = Critic(in_features=observation_space+action_space)
    critic_1_target = Critic(in_features=observation_space+action_space)
    critic_2_target = Critic(in_features=observation_space+action_space)

    actor = actor.to(device)
    actor_target = actor_target.to(device)
    critic_1 = critic_1.to(device)
    critic_2 = critic_2.to(device)
    critic_1_target = critic_1_target.to(device)
    critic_2_target = critic_2_target.to(device)

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

    agent.load(path, epoch)

    if environment == "dm_control":
        env = gym.make(env_name)
        state = env.reset()
        state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
        state = torch.from_numpy(state).float()
        viewer.launch(env, policy=agent.render_policy)

    elif environment == "gym":
        env = suite.load('cheetah', 'run')
        unplugged_rl_step_count = 0
        time_step = env.reset()
        state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
        state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
        state = torch.from_numpy(state).float()

        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            state = (state-agent.replay_buffer.mean)/agent.replay_buffer.std
            state = torch.from_numpy(state).float()
            env.render()
            time.sleep(0.01)
    env.close()

    return 

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

    parser.add_argument("env", help="The environment used")
    parser.add_argument("epoch", help="The epoch of the trained agent")
    parser.add_argument("path", help="The folder of the trained agent")
    
    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/td3+bc_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])
    
    params.fix()

    test(params, args.path, args.epoch, args.environment)
