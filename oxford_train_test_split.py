"""
Some random functions
"""

import os
import numpy as np


def create_train_test_split(
    data: list | np.ndarray,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    """Creates a train/val/test split given a list/array of datapoints"""
    
    # Splits must sum to 1.0
    assert(train_split + val_split + test_split == 1.0)
    
    print("%i data points." % data.shape)

    num_d = data.shape[0]
    indices = np.random.permutation(num_d)

    train_idx   = indices[                                     : int(num_d*train_split)]
    val_idx     = indices[              int(num_d*train_split) : int(num_d*(train_split+val_split))]
    test_idx    = indices[  int(num_d*(train_split+val_split)) : ]

    train_data  = data[train_idx]
    val_data    = data[val_idx]
    test_data   = data[test_idx]

    print("Split into %i train samples, %i val samples, %i test samples." % 
        (train_data.shape[0], val_data.shape[0], test_data.shape[0]))
    
    save_dir = os.path.join(os.path.dirname(data_file),'splits')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for filename, write_data in [('train.txt', train_data), ('val.txt', val_data), ('test.txt', test_data)]:
        with open(os.path.join(save_dir,filename), 'wt') as f:
            f.writelines(write_data)
    
    print("Saved splits to %s." %
          (save_dir))
            

if __name__ == "__main__":
    
    data_file = os.path.join(os.getcwd(),'data','oxford_pets','annotations','list.txt')
    with open(os.path.join(data_file), 'rt', encoding="utf-8") as f:
        data = f.readlines()
    data = data[6:]
    data = np.array(data)

    create_train_test_split(data)