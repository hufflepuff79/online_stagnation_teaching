import tensorflow as tf
import numpy as np
import pickle
import os
import argparse

def export_and_save_data(dir_in, dir_out, datapoints = None):

    num_shards = len(os.listdir(dir_in))

    filenames = [os.path.join(dir_in, f'train-{i:05d}-of-{num_shards:05d}') for i in range(num_shards)]

    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')

    if datapoints == None:
        datapoints = 0
        for e in raw_dataset:
            datapoints += 1

    data = {'observations' : np.zeros((datapoints, 17)),
            'next_observations' : np.zeros((datapoints, 17)),
            'rewards' : np.zeros(datapoints),
            'actions' : np.zeros((datapoints, 6)),
            'terminals' : np.zeros(datapoints)}

    c = 0
    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        sns_v = np.array(example.features.feature['observation/velocity'].float_list.value)
        sns_p = np.array(example.features.feature['observation/position'].float_list.value)
        data['observations'][c, :] = np.concatenate((sns_p[:8], sns_v[:9]))
        data['next_observations'][c, :] = np.concatenate((sns_p[8:], sns_v[9:]))
        data['actions'][c, :] = np.array(example.features.feature['action'].float_list.value)[:6]
        data['rewards'][c] = np.array(example.features.feature['reward'].float_list.value[0])
    c += 1

    a_file = open(os.path.join(dir_out, "data.pkl"), "wb")
    pickle.dump(data, a_file)
    a_file.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', type=str)
    parser.add_argument('dir_out', type=str)
    parser.add_argument('--datapoints', type=int, default=None)

    args = parser.parse_args()
    export_and_save_data(args.dir_in, args.dir_out, args.datapoints)
