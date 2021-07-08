import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D)
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

plt.style.use('seaborn-whitegrid')

### NETWORK PARAMETERS ###
DATA_SPLIT_PERC = 0.9
CHANNELS_CNN = 1
BATCH_SIZE = 8
EPOCHS = 150

### System ###
checkpoint_str = f"test/cp_ep{EPOCHS}_bsize{BATCH_SIZE}_adam.ckpt"
checkpoint_path = Path(checkpoint_str)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Make sure not to overwrite existing models
try:
    if len(os.listdir(checkpoint_path)) > 0:
        checkpoint_str += '_'
except FileNotFoundError:
    pass


# Load and shape the dataset
DATASETS = {0: ('sensor_readings_2.data', 2), 1: (
    'sensor_readings_4.data', 4), 2: ('sensor_readings_24.data', 24)}

for data in DATASETS:
    print(data, ':', DATASETS[data][0])

DATA_ID = int(input('Select dataset: '))
NUM_SENSORS = DATASETS[DATA_ID][1]
NUM_CLASSES = 4

df = pd.read_csv(f'Data/{DATASETS[DATA_ID][0]}', header=None)

# Encode the class labels
directions = {'Move-Forward': 0, 'Sharp-Right-Turn': 1,
              'Slight-Right-Turn': 2, 'Slight-Left-Turn': 3}
df.iloc[:, -1].replace(directions, inplace=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

(x_train, x_test), (y_train, y_test) = split(x, 90), split(y, 90)

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# The first parameter of 'shape' is the number of samples.
num_train_samples = x_train.shape[0]
num_batches = int(num_train_samples / BATCH_SIZE)

print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))


# Save the model every epoch, only if the model has improved
cp_callback = ModelCheckpoint(checkpoint_str,
                              monitor='accuracy',
                              save_best_only=True,
                              mode='max',
                              save_weights_only=False,
                              save_freq='epoch',
                              verbose=2)
callbacks = [cp_callback, tb_callback]


# Create a new model CNN

model = Sequential()

model.add(Conv1D(filters=128, kernel_size=1,
          activation='relu', input_shape=x_train.shape[1:]))

model.add(BatchNormalization())

model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))

model.add(BatchNormalization())

model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(128, activation='sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(NUM_CLASSES, activation='softmax'))
