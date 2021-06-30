import argparse

from matplotlib.pyplot import step
import torch
from utils import StatusPrinter, Parameters
from torch.utils.tensorboard import SummaryWriter
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al
import torch.nn as nn
import numpy as np
from os.path import exists, join
from os import makedirs

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params):

    # create a summary writer for logging stats
    log_dir = join("train_stats", str(int(time.time())))
    if not exists(log_dir):
        makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # create Atari game environment
    env = al.create_atari_environment(params.game, sticky_actions=params.env_sticky_actions)
    num_actions = env.action_space.n

    # create the Q network and Q target network
    Q_network = REM(num_actions=num_actions,
                    num_heads=params.model_num_heads, agent_history=params.agent_history)

    Q_target_network = REM(num_actions=num_actions,
                           num_heads=params.model_num_heads, agent_history=params.agent_history)

    optimizer = torch.optim.Adam(Q_network.parameters(), lr=params.adam_learning_rate, eps=params.adam_epsilon)

    # parallelism if multiple GPUs
    # if torch.cuda.device_count() > 1:
    #    Q_network = nn.DataParallel(Q_network)
    #    Q_target_network = nn.DataParallel(Q_target_network)

    # Network to GPU
    Q_network = Q_network.to(device)
    Q_target_network = Q_target_network.to(device)

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, num_actions, params.data_dir,
                     optimizer=optimizer, batch_size=params.replay_batch_size,
                     epsilon=params.agent_epsilon, gamma=params.agent_gamma, history=params.agent_history)

    # for logging
    sp = StatusPrinter()
    sp.add_counter("epoch", "Epochs", params.epochs, 0, bold=True)
    sp.add_bar("iter", "Iteration Progress", params.iterations)
    sp.add_bar("valid", "Validation Progress", params.validation_runs)

    # initiate training
    print(f"\nStarting Training\nEpochs: {params.epochs}\nIterations per Epoch: {params.iterations}\n\n")
    for epoch in range(params.epochs):

        sp.increment_and_print("epoch")
        sp.print_statement("iter")
        sp.reset_element("iter")

        for iteration in range(1, params.iterations + 1):

            sp.increment_and_print("iter")
            train_loss = agent.train_batch()

            if iteration % params.iter_target_update == 0:
                agent.update_target()

            if iteration % params.iter_buffer_update == 0:
                agent.replay_buffer.load_new_buffer()

        # online validation
        # set network status to eval
        agent.set_net_status(eval=True)
        # TODO: instead of playing to terminal state, play for certain amount of steps?
        sp.print_statement("valid")
        sp.reset_element("valid")

        total_reward = 0
        total_action_freq = np.zeros(num_actions)
        for run in range(params.validation_runs):
            sp.increment_and_print("valid")
            online_reward, action_freq = online_validation(agent=agent, env=env,
                                                           max_step_count=params.agent_max_val_steps)
            total_reward += online_reward
            total_action_freq += action_freq

        validation_reward = total_reward/params.validation_runs
        total_action_freq /= params.validation_runs

        writer.add_scalar('Validation/Avg_Reward', validation_reward, epoch)
        for i, freq in enumerate(total_action_freq):
            writer.add_scalar(f"ActionFrequency/A{i}", freq, epoch)
        print(f"Average Reward: {validation_reward}\n")


def online_validation(agent, env, max_step_count, render=False):

    step_count = 0
    total_reward = 0

    done = False
    state = env.reset()
    state = torch.from_numpy(state).float()
    state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
    agent.state_buffer.reset(state)

    freq_actions = np.zeros(env.action_space.n)

    while not done and step_count < max_step_count:
        action = agent.act(state, deterministic=False)
        state, reward, done, _ = env.step(action)
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        freq_actions[action] += 1

        if render:
            time.sleep(0.03)
            env.render("human")

        total_reward += reward
        step_count += 1

    freq_actions /= step_count
    return total_reward, freq_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_argument('--epochs', type=int, help='amount of epochs for training run')
    # TODO Goal: View 1.000.000 frames per epoch. --> problem: one iter (1 or 4) frames?
    parser.add_argument('--iterations', type=int, help='amount of iterations per epoch')
    parser.add_argument('--game', type=str, help='Atari game to train Agent on')
    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/paper_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    if args.data_dir:
        params.data_dir = args.data_dir
    if args.epochs:
        params.epochs = args.epochs
    if args.iterations:
        params.iterations = args.iterations
    if args.game:
        params.game = args.game

    params.fix()

    train(params)
