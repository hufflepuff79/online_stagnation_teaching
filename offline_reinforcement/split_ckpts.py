import argparse
import os
import subprocess
import numpy as np
import gzip
import glob

STORE_FILENAME_PREFIX = '$store$_'

ELEMS = ['observation', 'action', 'reward', 'terminal']

def load_and_split_gzips(inpath: str, outpath: str, num_split: int):
    num_files = len(glob.glob(os.path.join(inpath,'*action_ckpt*')))
    for i in range(num_files):
        # print(i)
        data = {}
        for elem in ELEMS:
            file_path = f'{inpath}{STORE_FILENAME_PREFIX}{elem}_ckpt.{i}.gz'
            file = gzip.open(file_path, 'rb')
            print(f'Loading {file_path}')
            data[elem] = np.load(file)
            data[elem] = np.array_split(data[elem], num_split)
            
            file.close()
            for j in range(num_split):
                file_outpath = f'{outpath}{STORE_FILENAME_PREFIX}{elem}_split_ckpt.{i * num_split + j}.gz'
                print(f'Writing {file_outpath}')
                outfile = gzip.open(file_outpath, 'wb')
                np.save(outfile, data[elem][j], allow_pickle=False)
        print(f"Finished splitting file number {i+1} of {num_files}")
        print("#########################")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
    parser.add_argument('--infolder', type=str, help='path to the folder with the gzips that should be splitted')
    parser.add_argument('--outfolder', type=str, help='path to the folder where the splitted gzips should be saved')
    parser.add_argument('--num_splits', type=int, default=4, help='1 folder should be splitted into this many equal splits')
    args = parser.parse_args()

    load_and_split_gzips(args.infolder, args.outfolder, args.num_splits)