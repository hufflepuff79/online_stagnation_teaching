import argparse
import torch
from utils import StatusPrinter
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al
import torch.nn as nn

import time

def train(atari_game, data_dir, epochs, iterations):

    # create Atari game environment
    env = al.create_atari_environment(atari_game)
    num_actions = env.action_space.n

    # create the Q network and Q target network
    Q_network = nn.DataParallel(REM(num_actions=num_actions))  #TODO: what values to give for num_actions, num_heads
    Q_target_network = nn.DataParallel(REM(num_actions=num_actions))  #TODO: what values to give for num_actions, num_heads

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, num_actions, data_dir)
    # agent.replay_buffer.load_new_buffer()

    # for logging
    sp = StatusPrinter()
    sp.add_counter("epoch", "Epochs", epochs, 0)
    sp.add_bar("iter", "Iteration Progress", iterations)

    # initiate training
    print(f"\nStarting Training\nEpochs: {epochs}\nIterations per Epoch: {iterations}\n\n")
    for epoch in range(epochs):
        
        sp.increment_and_print("epoch")
        sp.print_statement("iter")
        sp.reset_element("iter")

        iter_block = iterations//30

        for iteration in range(iterations):
            
            sp.increment_and_print("iter")
            loss_value = agent.train_batch()
            
            if iteration % 2000 == 0:
                agent.update_target()
                #agent.replay_buffer.load_new_buffer()

        # online validation
        # TODO: instead of playing to terminal state, play for certain amount of steps?
        validation_reward = online_validation(agent=agent, env=env)
        print(f"Average Reward: {validation_reward}\n")


        # TODO: track stats using tensorboard (needs to be added to Agent file)


def online_validation(agent, env, num_runs=5, render=False):
    total_reward = 0
    total_step_count = 0
    for _ in range(num_runs):
        step_count = 0
        done = False
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1,1,state.shape[0], state.shape[1]))
        agent.state_buffer.reset(state)

        while not done and step_count < 150000:
            action = agent.act(state, deterministic=False)
            state, reward, done, _ = env.step(action)  #TODO: does the input have to be a tuple
            state = torch.from_numpy(state).float()
            state = torch.reshape(state, (1,1, state.shape[0], state.shape[1]))
            if render:
                time.sleep(0.03)
                env.render("human")
            
            total_reward += reward
            step_count += 1
            total_step_count += 1

    return total_reward / total_step_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_argument('--epochs', type=int, help='amount of epochs for training run', default=200)
    # TODO Goal: View 1.000.000 frames per epoch. --> problem: one iter (1 or 4) frames?
    parser.add_argument('--iter', type=int, help='amount of iterations per epoch', default=8000)
    parser.add_argument('--game', type=str, help='Atari game to train Agent on', default='Breakout')
    args = parser.parse_args()

    train(atari_game=args.game,
          data_dir=args.data_dir,
          epochs=args.epochs,
          iterations=args.iter)
