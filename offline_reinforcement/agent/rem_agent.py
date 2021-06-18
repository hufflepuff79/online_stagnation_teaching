import tensorflow as tf
import torch
from agent.networks import REM
from agent.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class REMAgent:

    def __init__(self, Q, Q_target, data_dir):
        
        # setup networks
        self.Q = Q.to(device)
        self.Q_target = Q_target.to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # replay buffer
        self.replay_buffer = ReplayBuffer(data_dir)

        # parameters
        self.batch_size = 32 
        self.epsilon = 0.001
        self.gamma = 0.99

        # optimizer
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.00005, eps=0.0003125)

        # loss
        self.loss_function = torch.nn.SmoothL1Loss(beta=1.0)
    

    def train_batch(self):
        
        # sample replay buffer
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_done = self.replay_buffer.next_batch(self.batch_size)
         
        # random weights
        alphas = np.random.uniform(low=0, high=1, size=200)

        # update 
        max_action_Qs, _ = torch.max(self.Q_target(batch_next_states, alphas), dim=1)
        td_targets = batch_rewards + self.gamma * max_action_Qs * (1.0-batch_done)

        self.optimizer.zero_grad()
        Q_pred = self.Q(batch_states, alphas)[torch.arange(self.batch_size), batch_actions]
        loss = self.loss_function(Q_pred, td_targets)
        loss.backward()
        self.optimizer.step()


    def update_target(self):

        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, X_batch):
        raise NotImplementedError
