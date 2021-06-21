import argparse
import torch
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al

import time

def train(atari_game, data_dir, epochs):

    # create Atari game environment
    env = al.create_atari_environment(atari_game)
    num_actions = env.action_space.n

    # create the Q network and Q target network
    Q_network = REM(num_actions=num_actions)  #TODO: what values to give for num_actions, num_heads
    Q_target_network = REM(num_actions=num_actions)  #TODO: what values to give for num_actions, num_heads

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, num_actions, data_dir)

    # initiate training
    for epoch in range(epochs):
        for iteration in range(10):  #TODO: how many iterations per epoch?
            agent.train_batch()
        # TODO: when to update the Q target network
        agent.update_target()

        # TODO: environment setup and online validation
        validation_reward = online_validation(agent=agent, env=env)
        print(validation_reward)

        # TODO: track stats using tensorboard (needs to be added to Agent file)


def online_validation(agent, env, render=False):
    total_reward = 0
    step_count = 0
    done = False
    state = env.reset()
    state = torch.from_numpy(state).float()
    state = torch.reshape(state, (1,1,state.shape[0], state.shape[1]))
    agent.state_buffer.reset(state)

    while not done:
        action = agent.act(state, deterministic=False)
        state, reward, done, _ = env.step(action)  #TODO: does the input have to be a tuple
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1,1, state.shape[0], state.shape[1]))
        if render:
            time.sleep(0.03)
            env.render("human")
        
        total_reward += reward
        step_count += 1
    return total_reward / step_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_argument('--epochs', type=int, help='amount of epochs for training run', default=10)
    parser.add_argument('--game', type=str, help='Atari game to train Agent on', default='Breakout')

    # TODO: additional argument for how often validation is performed/Q_target update is performed?
    args = parser.parse_args()

    train(atari_game=args.game,
          data_dir=args.data_dir,
          epochs=args.epochs)
