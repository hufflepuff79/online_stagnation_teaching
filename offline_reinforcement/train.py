import argparse
from agent.rem_agent import REMAgent
from agent.networks import REM


def train(data_dir, epochs):

    # create the Q network and Q target network
    Q_network = REM()  #TODO: what values to give for num_actions, num_heads
    Q_target_network = REM()  #TODO: what values to give for num_actions, num_heads

    # create the REM Agent
    agent = REMAgent(Q_network, Q_target_network, data_dir)

    # initiate training
    for epoch in range(epochs):
        for iteration in range(iterations):  #TODO: how many iterations per epoch?
            agent.train_batch()
        # TODO: when to update the Q target network
        agent.update_target()

        # TODO: environment setup and online validation
        agent.act()

        # TODO: track stats using tensorboard (needs to be added to Agent file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--data_dir', type=str, help='location of training data')
    parser.add_arguement('--epochs', type=int, help='amount of epochs for training run', default=100)
    args = parser.parse_args()
    train(args.data_dir, args.epochs)