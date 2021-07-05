import torch.nn as nn
import torch
import torch.nn.functional as F


class REM(nn.Module):

    def __init__(self, num_actions: int = 18, num_heads: int = 200, agent_history: int = 4):
        super(REM, self).__init__()

        self.num_heads = num_heads

        self.conv1 = nn.Conv2d(in_channels=agent_history, out_channels=32, kernel_size=8, stride=4, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')

        self.lin1 = nn.Linear(in_features=3136, out_features=512)

        self.heads = nn.ModuleList([nn.Linear(in_features=512, out_features=num_actions) for i in range(num_heads)])

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor, alphas: list):
        """
        Forward pass of REM-Network

        x: Input Batch
        alphas: weights of random mixture
        """

        if len(alphas) != self.num_heads:
            raise ValueError("weights need too be of same length as network heads")
        
        x = torch.div(x, 255.0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)

        return sum(alpha*lin(x) for lin, alpha in zip(self.heads, alphas))
