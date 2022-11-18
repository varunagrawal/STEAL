import os
import numpy as np
from lasa_load_data import LASA

def main():
    
    print('Loading dataset...')
    
    data_path = os.getcwd()+'/../data/LASADataset'
    lasa = LASA(data_path, data_name='heee')
    
    num_demos = lasa.num_demos
    dataset = lasa.trajectory
    
    for i in range(num_demos):
        
        pos = np.array(dataset[i][0])
        time = np.array(dataset[i][1])
        vel = np.array(dataset[i][2])
        acc = np.array(dataset[i][3])
    
    

if __name__ == "__main__":
    main()
