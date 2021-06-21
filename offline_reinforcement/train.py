import argparse
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al


def train(atari_game, data_dir, epochs):

    # create Atari game environment
    env = al.create_atari_environment(atari_game)

    # create the Q network and Q target network
    Q_network = REM()  #TODO: what values to give for num_actions, num_heads
    Q_target_network = REM()  #TODO: what values to give for num_actions, num_heads

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, data_dir)

    # initiate training
    for epoch in range(epochs):
        for iteration in range(1):  #TODO: how many iterations per epoch?
            agent.train_batch()
        # TODO: when to update the Q target network
        agent.update_target()

        # TODO: environment setup and online validation
        validation_reward = online_validation(agent=agent, env=env)

        # TODO: track stats using tensorboard (needs to be added to Agent file)


def online_validation(agent, env):
    total_reward = 0
    step_count = 0
    state, reward, done, _ = env.reset()

    while not done:
        action = agent.act(state, deterministic=False)
        state, reward, done, _ = env.step(action)  #TODO: does the input have to be a tuple
        total_reward += reward
        step_count += 1
    return total_reward / step_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_arguement('--epochs', type=int, help='amount of epochs for training run', default=1)
    parser.add_argument('--game', type=str, help='Atari game to train Agent on', default='Pong-v0')

    # TODO: additional argument for how often validation is performed/Q_target update is performed?
    args = parser.parse_args()

    train(atari_game=args.game,
          data_dir=args.data_dir,
          epochs=args.epochs)