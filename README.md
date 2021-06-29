# Deep Learning Lab Final Project:
# Online Stagnation Teaching

In this project we aim to implement the Offline Reinforcement Learning approach as presented in the paper "*An Optimistic Perspective on Offline Reinforcement Learning*" by Agarwal et al. We train our agents on the classic Atari games Asterix, Breakout, Pong, Q\*bert, and Seaquest. While the paper uses TensorFlow for their implementation, ours is based on PyTorch.

# Local Setup
To run our project locally you will need to install the following:
 - Dopamine (install via source [directions](https://github.com/google/dopamine#install-via-source))
 - PyTorch

# Starting a Training Run
To start a training run of the REM Agent, execute the command `python train.py`. 

Additionally it is possible to specify the optional arguments *data_dir*, *epochs*, *iteration*, *game*, and *cfg* via the command line (ex `python train --game Asterix`). You can use execute `python train.py --help` to learn more about these arguments. Additional configurations for training are defined and can be changed in the configuration files found in the [parameters files folder](offline_reinforcement/parameter_files/) (see section Training Configurations).


# Training Configurations
The default training configurations based on the "An Optimistic Perspective on Offline Reinforcement Learning" by Agarwal et al. can be found in [parameters files folder](offline_reinforcement/parameter_files/). Please insure that if using multiple GPUs for training, the *replay_batch_size* parameter is divisible by the number of GPUs in use. 
