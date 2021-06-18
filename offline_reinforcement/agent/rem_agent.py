import torch
from agent.networks import REM

class REMAgent:

    def __init__(self):
        raise NotImplementedError

    def update(self, X_batch, y_batch):
        raise NotImplementedError

    def validate(self, X_batch, y_batch):
        raise NotImplementedError

    def act(self, X_batch):
        raise NotImplementedError

    def load(self, file_name):
        raise NotImplementedError

    def save(self, file_name):
        raise NotImplementedError
