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
import wandb

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params):


    # create a summary writer for logging stats
    log_dir = join("train_stats", str(int(time.time())))
    if not exists(log_dir):
        makedirs(log_dir)
    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb.init(entity="online_stagnation_teaching")
    writer = SummaryWriter(log_dir=log_dir)
    
    writer.add_text('Parameters', str(params))

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
                     epsilon=params.agent_epsilon, gamma=params.agent_gamma,
                     history=params.agent_history, suffix=params.fixed_checkpoint)

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

        train_loss = 0

        for iteration in range(1, params.iterations + 1):
            train_loss += agent.train_batch()

            if iteration % params.iter_target_update == 0:
                agent.update_target()

            if iteration % params.iter_buffer_update == 0:
                agent.replay_buffer.load_new_buffer(suffix=params.fixed_checkpoint)

            sp.increment_and_print("iter")

        train_loss /= params.iterations
       
        writer.add_scalar('Training/Avg_Loss', train_loss, epoch)
        print(f"Average Training Loss: {train_loss}")

        # online validation
        agent.set_net_status(eval=True)
        # TODO: instead of playing to terminal state, play for certain amount of steps?
        sp.print_statement("valid")
        sp.reset_element("valid")

        total_reward = 0
        total_action_freq = np.zeros(num_actions)
        for run in range(params.validation_runs):
            online_reward, action_freq = online_validation(agent=agent, env=env,
                                                           max_step_count=params.agent_max_val_steps)
            total_reward += online_reward
            total_action_freq += action_freq
            sp.increment_and_print("valid")

        validation_reward = total_reward/params.validation_runs
        total_action_freq /= params.validation_runs

        for i, freq in enumerate(total_action_freq):
            writer.add_scalar(f"ActionFrequency/A{i}", freq, epoch)
        
        writer.add_scalar('Validation/Avg_Reward', validation_reward, epoch)
        print(f"Average Reward: {validation_reward}\n")

        # save weights at regular intervals
        if epoch % params.agent_save_weights == 0:
            weights_file = join(log_dir, "agent_weights_epoch_{}.pth".format(epoch))
            agent.save(weights_file)


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
    parser.add_argument('--game', type=str, help='Name of the atari environment')
    parser.add_argument('--data_dir', type=str, help='Path to the transition data')
    parser.add_argument('--epochs', type=int, help='Number of test evaluations. Every epoch there are <iterations> iterations')
    parser.add_argument('--iterations', type=int, help='aNumber of iterations per epoch. Agent gets and trains one batch per epoch')
    parser.add_argument('--validation_runs', type=int, help='How many runs to make for test evaluation. Average over runs is returned')
    parser.add_argument('--iter_target_update', type=int, help='Every <iter_target_update> iterations, the target net is updated')
    parser.add_argument('--iter_buffer_update', type=int, help='Every <iter_buffer_update> iteraions, a new buffer is loaded from the transition data')
    parser.add_argument('--adam_learning_rate', type=float, help='Learning rate of ADAM optimizer')
    parser.add_argument('--adam_epsilon', type=float, help='Stabilization term of ADAM optimizer')
    parser.add_argument('--model_num_heads', type=int, help='Number of heads of the REM')  
    parser.add_argument('--agent_epsilon', type=float, help='Epsilon-greedy parameter for acting of the agent')
    parser.add_argument('--agent_gamma', type=float, help='Discount factor of the REM agent')
    parser.add_argument('--agent_history', type=int, help='Number of states which are stacked as a multi-channel image in one go into the REM')
    parser.add_argument('--agent_max_val_steps', type=int, help='Maximum steps per evaluation run of agent, if not terminated before.')
    parser.add_argument('--replay_batch_size', type=int, help='Batch size for training the agent with the transition data')
    parser.add_argument('--env_sticky_actions', type=bool, help='If sticky actions should be used in online validation')
    parser.add_argument("--agent_save_weights", type=int, help="Frequency at which the weights of network are saved")
    parser.add_argument("--fixed_checkpoint", type=int, help="Fixed checkpoint number to debug. Default is None for random checkpoint")
    
    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/paper_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])
    
    params.fix()

    train(params)
