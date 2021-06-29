import argparse
import torch
from utils import StatusPrinter, Parameters
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al
import torch.nn as nn

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(atari_game, data_dir, epochs, iterations,
          validation_runs=5,
          iter_target_update=2000,
          iter_buffer_update=2000,
          adam_learning_rate=0.00005, 
          adam_epsilon=0.0003125,
          model_num_heads=200,
          agent_epsilon=0.001, 
          agent_gamma=0.99,
          agent_history=4,
          replay_batch_size=32):

    # create Atari game environment
    env = al.create_atari_environment(atari_game)
    num_actions = env.action_space.n

    # create the Q network and Q target network
    Q_network = REM(num_actions=num_actions, num_heads=model_num_heads)  #TODO: what values to give for num_actions, num_heads
    Q_target_network = REM(num_actions=num_actions, num_heads=model_num_heads)  #TODO: what values to give for num_actions, num_heads
    optimizer = torch.optim.Adam(Q_network.parameters(), lr=adam_learning_rate, eps=adam_epsilon)

    # parallelism if multiple GPUs
    #if torch.cuda.device_count() > 1:
    #    Q_network = nn.DataParallel(Q_network)
    #    Q_target_network = nn.DataParallel(Q_target_network)

    # Network to GPU
    Q_network = Q_network.to(device)
    Q_target_network = Q_target_network.to(device)

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, num_actions, data_dir, optimizer=optimizer, batch_size=replay_batch_size, epsilon=agent_epsilon, gamma=agent_gamma, history=agent_history)
    agent.replay_buffer.load_new_buffer()

    # for logging
    sp = StatusPrinter()
    sp.add_counter("epoch", "Epochs", epochs, 0, bold=True)
    sp.add_bar("iter", "Iteration Progress", iterations)
    sp.add_bar("valid", "Validation Progress", validation_runs)

    # initiate training
    print(f"\nStarting Training\nEpochs: {epochs}\nIterations per Epoch: {iterations}\n\n")
    for epoch in range(epochs):
        
        sp.increment_and_print("epoch")
        sp.print_statement("iter")
        sp.reset_element("iter")

        for iteration in range(iterations):
            
            sp.increment_and_print("iter")
            loss_value = agent.train_batch()
            
            if iteration % iter_target_update == 0:
                agent.update_target()
            
            if iteration % iter_buffer_update == 0:
                agent.replay_buffer.load_new_buffer()

        # online validation
        # TODO: instead of playing to terminal state, play for certain amount of steps?
        total_reward = 0
       
        sp.print_statement("valid")
        sp.reset_element("valid")
        for run in range(validation_runs):
            sp.increment_and_print("valid")
            total_reward += online_validation(agent=agent, env=env)
            
        validation_reward = total_reward/validation_runs
        print(f"Average Reward: {validation_reward}\n")


        # TODO: track stats using tensorboard (needs to be added to Agent file)


def online_validation(agent, env, max_step_count=1500, render=False):

    step_count = 0
    total_reward = 0
    
    done = False
    state = env.reset()
    state = torch.from_numpy(state).float()
    state = torch.reshape(state, (1,1,state.shape[0], state.shape[1]))
    agent.state_buffer.reset(state)

    while not done and step_count < max_step_count:
        action = agent.act(state, deterministic=False)
        state, reward, done, _ = env.step(action)  #TODO: does the input have to be a tuple
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        if render:
            time.sleep(0.03)
            env.render("human")
        
        total_reward += reward
        step_count += 1

    return total_reward / step_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_argument('--epochs', type=int, help='amount of epochs for training run', default=200)
    # TODO Goal: View 1.000.000 frames per epoch. --> problem: one iter (1 or 4) frames?
    parser.add_argument('--iter', type=int, help='amount of iterations per epoch', default=8000)
    parser.add_argument('--game', type=str, help='Atari game to train Agent on', default='Breakout')
    parser.add_argument('--cfg', type=str, help='path to json config file', default=None)
    args = parser.parse_args()

    if args.cfg:
        param = Parameters(args.cfg)
        train(atari_game=param.game,
              data_dir=param.data_dir,
              epochs=param.epochs,
              iterations=param.iterations,
              validation_runs=param.validation_runs,
              iter_target_update=param.iter_target_update,
              iter_buffer_update=param.iter_buffer_update,
              adam_learning_rate=param.adam_learning_rate,
              adam_epsilon=param.adam_epsilon,
              model_num_heads=param.model_num_heads,
              agent_epsilon=param.agent_epsilon,
              agent_gamma=param.agent_gamma,
              agent_history=param.agent_history,
              replay_batch_size=param.replay_batch_size)
    
    else:
        train(atari_game=args.game,
              data_dir=args.data_dir,
              epochs=args.epochs,
              iterations=args.iter)
