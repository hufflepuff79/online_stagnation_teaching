import tensorflow as tf
import numpy as np
import pickle
import os
import argparse
from dm_control import suite


def export_and_save_data(dir_in, dir_out, env_name, task):

    env = suite.load(env_name, task)
    time_step = env.reset()
    obs_names = list(time_step.observation.keys())
    obs_dims = [(len(x) if isinstance(x, np.ndarray) else 1) for x in time_step.observation.values()]
    act_dim = env.action_spec().shape[0]

    num_shards = len(os.listdir(dir_in))

    filenames = [os.path.join(dir_in, f'train-{i:05d}-of-{num_shards:05d}') for i in range(num_shards)]

    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')

    datapoints = 0
    for e in raw_dataset:
        datapoints += 1

    data = {'observations': np.zeros((datapoints, sum(obs_dims)), dtype=np.float32),
            'next_observations': np.zeros((datapoints, sum(obs_dims)), dtype=np.float32),
            'rewards': np.zeros(datapoints, dtype=np.float32),
            'actions': np.zeros((datapoints, act_dim), dtype=np.float32),
            'terminals': np.zeros(datapoints, dtype=np.bool)}

    c = 0
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        length = 0
        for name, dim in zip(obs_names, obs_dims):
            obs = np.array(example.features.feature[f'observation/{name}'].float_list.value)
            data['observations'][c, length:length+dim] = obs[:dim]
            data['next_observations'][c, length:length+dim] = obs[dim:]
            length += dim
        data['actions'][c, :] = np.array(example.features.feature['action'].float_list.value)[:act_dim]
        data['rewards'][c] = np.array(example.features.feature['reward'].float_list.value[0])
        c += 1

    a_file = open(os.path.join(dir_out, f"{env_name}_data.pkl"), "wb")
    pickle.dump(data, a_file)
    a_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', type=str)
    parser.add_argument('dir_out', type=str)
    parser.add_argument('--name', type=str, default='cheetah')
    parser.add_argument('--task', type=str, default='run')

    args = parser.parse_args()
    export_and_save_data(args.dir_in, args.dir_out, args.name, args.task)
