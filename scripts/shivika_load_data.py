import os
import numpy as np
from lasa_load_data import LASA
import pickle as pkl
from pathlib import Path

# script for generating .pickle files from .mat files
def main():
    
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path,'data/LASADataset')

    file_list = []
    for file in os.listdir(data_path):
        if file.endswith('.mat'):
            file_list.append(file)
    
    for file in file_list:
        data_name = file
        lasa = LASA(data_path, data_name=data_name)
        dataset = lasa.trajectory
        dump_path = os.path.join(dir_path,'data/LASADataset_traj_pkl')

        with open(os.path.join(dump_path,os.path.splitext(file)[0]+'.pickle'), 'wb') as handle:
            pkl.dump(dataset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    

if __name__ == "__main__":
    main()
