import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio


class LASA:
    ''''
    Loads LASA dataset
    path: Path for the directory 
    data_name: .mat file name 
    returns trajectory: List of 7 demonstrations
    
    trajectory format: List of 7 demonstrations. Each demonstration has 
    following components:
        * (2x1000) array: x & y positions
        * (1x1000) array: time
        * (2x1000) array: x & y velocity
        * (2x1000) array: x & y acceleration

    '''

    def __init__(self, path, data_name) :
        self.trajectory = self.load_data(path, data_name) 


    def load_data(self, path, data_name):
        dataset_path = path
        data = sio.loadmat(os.path.join(dataset_path, data_name))
        
        trajectories = []

        dataset = data['demos']
        num_demos = int(dataset.shape[1])
        
        for i in range(num_demos):
            
            demo = []
            pos_ = dataset[0, i]['pos'][0][0]
            demo.append(pos_)

            time_ = dataset[0,i]['t'][0][0]
            demo.append(time_)

            vel_ = dataset[0, i]['vel'][0][0]
            demo.append(vel_)

            acc_ = dataset[0, i]['acc'][0][0]
            demo.append(acc_)

            trajectories.append(demo)

        return trajectories