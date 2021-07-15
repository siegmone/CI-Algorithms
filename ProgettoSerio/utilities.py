import h5py
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical

#from tensorflow.keras.utils import to_categorical

data = h5py.File('unscaled_data.h5', 'r')

# split data func


def split_data(dataset):
    return dataset['train'], dataset['valid'], dataset['test']


def to_array(dataset):
    arr = np.zeros(dataset.shape)
    dataset.read_direct(arr)
    return arr


et, hl, ht, mass, y = data['et'], data['hl'], data['ht'], data['mass'], data['y']
et_train, et_valid, et_test = split_data(et)
hl_train, hl_valid, hl_test = split_data(hl)
ht_train, ht_valid, ht_test = split_data(ht)
mass_train, mass_valid, mass_test = split_data(mass)
y_train, y_valid, y_test = split_data(y)
y_test_cat = to_categorical(y_test)

print(y_test_cat[0])

et_im = to_array(et_train)
img = Image.fromarray(et_im[0, :, :, 0], 'L')
img.show()
