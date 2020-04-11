import h5py
from config import config
import numpy as np
import matplotlib.pyplot as plt

def get_data(size=None, batch_size=32, shuffle=True):
    with h5py.File(config['dataset_original_balanced'], 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]#.astype('int32')
        X_val = hf['X_val'][:]
        y_val = hf['y_val'][:]#.astype('int32')
    np.random.seed(321)
    if shuffle == True:
        perm = np.random.permutation(y_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]
        perm = np.random.permutation(y_val.shape[0])
        X_val = X_val[perm]
        y_val = y_val[perm]
    if size is not None:
        X_train = X_train[:size]
        y_train = y_train[:size]
        X_val = X_val[:int(0.2 * size / 0.8)]
        y_val = y_val[:int(0.2 * size / 0.8)]
#     y_train = np.where(y_train == -1, 3, y_train)
#     y_val = np.where(y_val == -1, 3, y_val)
    plt.figure()
    plt.hist(y_train.reshape((-1, 1)), alpha=0.5, label='y_train')
    plt.hist(y_val.reshape((-1, 1)), alpha=0.5, label='y_val')
    plt.legend()
    plt.savefig('catogory_distribution.png')
    #print(X_train.shape)
    #assert 0
#     X_train = X_train.reshape((batch_size, -1, 600, 64))
#     y_train = y_train.reshape((batch_size, -1, 600, 1))
#     X_val = X_val.reshape((batch_size, -1, 600, 64))
#     y_val = y_val.reshape((batch_size, -1, 600, 1))
    return X_train, y_train, X_val, y_val
