import argparse
import torch
from utils import StatusPrinter, Parameters
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


def train(params, log_wb: bool = False, logging_freq: int = 1000):

    # create wandb for logging stats
    if log_wb:
        wandb.init(entity="online_stagnation_teaching", config=params.as_dict())

    # directory to log agent parameters
    log_dir = join("train_stats", wandb.run.name if log_wb else str(int(time.time())))
    if not exists(log_dir):
        makedirs(log_dir)

    # create Atari game environment
    env = al.create_atari_environment(params.game, sticky_actions=params.env_sticky_actions)
    action_names = env.environment.unwrapped.get_action_meanings()

    num_actions = env.action_space.n

    # create the Q network and Q target network
    Q_network = REM(num_actions=num_actions,
                    num_heads=params.model_num_heads, agent_history=params.agent_history)

    Q_target_network = REM(num_actions=num_actions,
                           num_heads=params.model_num_heads, agent_history=params.agent_history)

    optimizer = torch.optim.Adam(Q_network.parameters(), lr=params.adam_learning_rate, eps=params.adam_epsilon)

    # Network to GPU
    Q_network = Q_network.to(device)
    Q_target_network = Q_target_network.to(device)

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, num_actions, params.data_dir,
                     optimizer=optimizer, batch_size=params.replay_batch_size,
                     epsilon=params.agent_epsilon, gamma=params.agent_gamma,
                     history=params.agent_history, suffixes=params.fixed_checkpoint,
                     n_ckpts=params.n_ckpts)

    if params.agent_state_dict:
        agent.load(params.agent_state_dict)

    # for logging
    sp = StatusPrinter()
    sp.add_counter("epoch", "Epochs", params.epochs, 0, bold=True)
    sp.add_bar("iter", "Iteration Progress", params.iterations)
    sp.add_bar("valid", "Validation Progress", params.agent_total_steps + params.agent_episode_max_steps)
    logging = False

    # initiate training
    print(f"\nStarting Training\nEpochs: {params.epochs}\nIterations per Epoch: {params.iterations}\n\n")

    # Use the smaller split ckpts or not
    try:
        load_new_buffer_params = {'suffixes': params.fixed_checkpoint,
                                  'use_splits': True if params.num_split != 0 else False,
                                  'max_suffix': params.num_split}
    except Exception:
        load_new_buffer_params = {'suffixes': params.fixed_checkpoint}

    for epoch in range(params.epochs):
        sp.increment_and_print("epoch")
        sp.print_statement("iter")
        sp.reset_element("iter")

        train_loss = 0

        for iteration in range(params.iterations):

            if iteration % params.iter_buffer_update == 0:
                agent.replay_buffer.load_new_buffer(**load_new_buffer_params)

            # check if logging
            if iteration % logging_freq == 0 and log_wb:
                logging = True

            # train
            loss, log_dict = agent.train_batch(logging)
            train_loss += loss

            # log the results
            if logging:
                wandb.log({**log_dict, **{'epoch': epoch}})
                logging = False

            # update the target
            if (iteration+1) % params.iter_target_update == 0:
                agent.update_target()

            sp.increment_and_print("iter")

        sp.done_element("iter")
        train_loss /= params.iterations

        if log_wb:
            wandb.log({'Training/Avg_Loss': train_loss, 'epoch': epoch})
        print(f"Average Training Loss: {train_loss}")

        # online validation
        agent.set_net_status(eval=True)

        sp.print_statement("valid")
        sp.reset_element("valid")

        average_reward, action_freq = online_validation(agent=agent, env=env,
                                                        total_steps=params.agent_total_steps,
                                                        episode_max_steps=params.agent_episode_max_steps,
                                                        status_func=sp.increment_and_print,
                                                        status_arg="valid")
        sp.done_element("valid")

        if log_wb:
            for i, freq in enumerate(action_freq):
                wandb.log({f'ActionFrequency/A{i}: {action_names[i]}': freq, 'epoch': epoch})

            wandb.log({'Validation/Avg_Reward': average_reward, 'epoch': epoch})
        print(f"Average Reward: {average_reward}\n")

        # save weights at regular intervals
        if epoch % params.agent_save_weights == 0:
            weights_file = join(log_dir, "agent_weights_epoch_{}.pth".format(epoch))
            agent.save(weights_file)


def online_validation(agent, env, total_steps, episode_max_steps,
                      status_func=lambda *args: None, status_arg=None, render=False):

    # total step count
    tsc = 0
    total_reward = 0
    num_episodes = 0
    total_freq_actions = np.zeros(env.action_space.n)

    # only start a new episode if step count is below total steps
    while tsc < total_steps:

        # episode step count
        esc = 0
        done = False

        # reset environment
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        agent.state_buffer.reset(state)

        freq_actions = np.zeros(env.action_space.n)

        # either episode ends naturally or is terminated after max steps
        while not done and esc < episode_max_steps:
            action = agent.act(state, deterministic=False)
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state).float()
            state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
            freq_actions[action] += 1

            if render:
                time.sleep(0.03)
                env.render("human")

            total_reward += reward
            esc += 1
            tsc += 1
            status_func(status_arg)

        num_episodes += 1
        total_freq_actions += freq_actions

    total_freq_actions /= tsc
    total_reward /= num_episodes

    return total_reward, total_freq_actions


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
    parser.add_argument('--agent_total_steps', type=int, help='Total steps per evaluation run of agent. If over this value no new episode is started.')
    parser.add_argument('--agent_episode_max_steps', type=int, help='Maximum steps per episode of agent, if not terminated before.')
    parser.add_argument('--agent_state_dict', type=str, help='Path to the saved weights of an agent.')
    parser.add_argument('--replay_batch_size', type=int, help='Batch size for training the agent with the transition data')
    parser.add_argument('--env_sticky_actions', type=bool, help='If sticky actions should be used in online validation')
    parser.add_argument("--agent_save_weights", type=int, help="Frequency at which the weights of network are saved")
    parser.add_argument("--fixed_checkpoint", type=int, help="Fixed checkpoint number to debug. Default is None for random checkpoint")
    parser.add_argument("--n_ckpts", type=int, help="Number of checkpoints loaded in replay buffer at once")
    parser.add_argument("--num_split", type=int, default=0, help="Only set if you want to use the split checkpoints. Then set this parameter to the max index of a split_ckpt")
    parser.add_argument("--wandb", action='store_true', help="Log with wandb")

    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/paper_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])

    params.fix()

    train(params, log_wb=args.wandb)
