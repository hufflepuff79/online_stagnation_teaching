# Deep Learning Lab Final Project:
# Online Stagnation Teaching

In this project we aim to implement the Offline Reinforcement Learning approach as presented in the paper "An Optimistic Perspective on Offline Reinforcement Learning" by Agarwal et al. We train our agents on the classic Atari games Asterix, Breakout, Pong, Q\*bert, and Seaquest. While the paper uses TensorFlow for their implementation, ours is based on PyTorch.

# Local Setup
To run our project locally you will need to install the following:
 - Dopamine (install via source: https://github.com/google/dopamine#install-via-source)
 - PyTorch

# 


# Training Configurations
The default training configurations based on the "An Optimistic Perspective on Offline Reinforcement Learning" by Agarwal et al. can be found in [parameters files folder](offline_reinforcement/parameter_files/). Please insure that if using multiple GPUs for training, the *replay_batch_size* parameter is divisible by the number of GPUs in use. 
