import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

data = h5py.File('data.h5', 'r')

# split data func


def split_data(dataset):
    return dataset['train'], dataset['valid'], dataset['test']


et, hl, ht, mass, y = data['et'], data['hl'], data['ht'], data['mass'], data['y']
et_train, et_valid, et_test = split_data(et)
hl_train, hl_valid, hl_test = split_data(hl)
ht_train, ht_valid, ht_test = split_data(ht)
mass_train, mass_valid, mass_test = split_data(mass)
y_train, y_valid, y_test = split_data(y)

arr = np.zeros(y_train.shape)
y_train.read_direct(arr)
print(to_categorical(arr))
print(y_train.shape)
