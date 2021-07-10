import argparse
from matplotlib.pyplot import step
import torch
from utils import StatusPrinter, Parameters
from agent.rem_agent import REMAgent
from agent.networks import REM
from dopamine.discrete_domains import atari_lib as al
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(params, weights):
    # create the atari environment
    env = al.create_atari_environment(params.game, sticky_actions=False)
    num_actions = env.action_space.n

    # create the Q network and load in the weights
    Q_network = REM(num_actions=num_actions,
                    num_heads=params.model_num_heads,
                    agent_history=params.agent_history)

    Q_network.load_state_dict(torch.load(weights))
    Q_network = Q_network.to(device)
    Q_target = Q_network.deepcopy()

    agent = REMAgent(Q_network, Q_target, num_actions, params.data_dir, optimizer=None)
    agent.set_net_status(eval=True)

    while True:
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))
        agent.state_buffer.reset(state)
        done = False

        while not done:
            action = agent.act(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state).float()
            state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))

            time.sleep(0.03)
            env.render("human")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument("load_weights", type=str, help="location of weights to be tested")

    parser.add_argument('--game', type=str, help='Name of the atari environment')
    parser.add_argument('--data_dir', type=str, help='Path to the transition data')

    parser.add_argument('--cfg', type=str, help='path to json config file',
                        default='parameter_files/paper_parameters.json')
    args = parser.parse_args()

    params = Parameters(args.cfg)

    params.overload(args, ignore=['cfg'])

    params.fix()

    test(params, args.load_weights)
