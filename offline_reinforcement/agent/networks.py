import torch.nn as nn
import torch
import torch.nn.functional as F

class REM(nn.Module):

    def __init__(self, num_actions=18):
        super(REM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin1 = nn.Linear(in_features=3136, out_features=512)
        
        self.heads = [nn.Linear(in_features=512, out_features=num_actions) for i in range(200)]

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)

        return [lin(x) for lin in self.heads]
