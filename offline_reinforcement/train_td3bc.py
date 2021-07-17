import argparse
from matplotlib.pyplot import step
import torch
from utils import StatusPrinter, Parameters
from torch.utils.tensorboard import SummaryWriter
from agent.tdr3bc_agent import TDR3BC
from agent.networks import Actor, Critic
import torch.nn as nn
import numpy as np
from os.path import exists, join
from os import makedirs
import wandb

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(params, log_wb: bool = False, logging_freq: int = 1000):

    # Logging
    # create wandb for logging stats
    if log_wb:
        wandb.init(entity="online_stagnation_teaching", config=params.as_dict())

    # directory to log agent parameters
    log_dir = join("train_stats", wandb.run.name if log_wb else str(int(time.time())))
    if not exists(log_dir):
        makedirs(log_dir)

    # create the environment

    # create networks TODO: add the parameters for the networks
    actor = Actor()
    actor_target = Actor()

    critic_1 = Critic()
    critic_2 = Critic()
    critic_1_target = Critic()
    critic_2_target = Critic()

    actor = actor.to(device)
    actor_target = actor_target.to(device)
    critic_1 = critic_1.to(device)
    critic_2 = critic_2.to(device)
    critic_1_target = critic_1_target.to(device)
    critic_2_target = critic_2_target.to(device)

    # create optimizers #TODO: check parameters
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=params.adam_learning_rate, eps=params.adam_epsilon)
    critic_1_optimizer = optimizer = torch.optim.Adam(critic_1.parameters(), lr=params.adam_learning_rate,
                                                      eps=params.adam_epsilon)
    critic_2_optimizer = optimizer = torch.optim.Adam(critic_2.parameters(), lr=params.adam_learning_rate,
                                                      eps=params.adam_epsilon)


    # create the TD3+BC agent
    agent = TDR3BC(actor, actor_target, critic_1, critic_1_target,critic_2, critic_2_target,
                   actor_optimizer, critic_1_optimizer, critic_2_optimizer)

