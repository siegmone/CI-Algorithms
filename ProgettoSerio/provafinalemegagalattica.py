import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from helper_functions import *
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam
from tensorflow.keras.utils import to_categorical

plt.style.use('seaborn-whitegrid')


data = h5py.File('data.h5', 'r')

et, hl, ht, mass, y = data['et'], data['hl'], data['ht'], data['mass'], data['y']
et_train, et_valid, et_test = get_img_data(et)
ht_train, ht_valid, ht_test = get_img_data(ht)
hl_train, hl_valid, hl_test = get_data(hl)
mass_train, mass_valid, mass_test = get_data(mass)
y_train, y_valid, y_test = get_data(y)

NUM_CLASSES = 2

et_input_shape = define_inputshape(et_train)
et_width, et_height, et_channels = et_input_shape
et_train_r, et_valid_r, et_test_r = shape_data(et_train, et_valid, et_test)

ht_input_shape = define_inputshape(ht_train)
ht_width, ht_height, ht_channels = ht_input_shape
ht_train_r, ht_valid_r, ht_test_r = shape_data(ht_train, ht_valid, ht_test)

y_train_cat = to_categorical(y_train)
y_valid_cat = to_categorical(y_valid)
y_test_cat = to_categorical(y_test)


cnn_et = create_cnn(input_shape=et_input_shape, filters=(
    16, 32, 64, 128), n_dense=3, hidden_units=128, dp=0.1)
# cnn_ht = create_cnn(width=ht_width, height=ht_height, depth=ht_channels, filters=(
#     27, 27), n_dense=2, hidden_units=84, dp=0.5)
model = cnn_et

# combinedInput = concatenate([cnn_et.output, cnn_ht.output])

# x = Dense(32, activation="relu")(combinedInput)


# model = Model(inputs=[cnn_et.input, cnn_ht.input], outputs=x)


opt = Adam(epsilon=0.01, learning_rate=0.000001)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy'])


# history = model.fit(x=[et_train_r, ht_train_r], y=y_train_cat, batch_size=128,
#                     epochs=100, verbose=1, validation_data=([et_valid_r, ht_valid_r], y_valid_cat))

history = model.fit(x=et_train_r, y=y_train_cat, batch_size=128,
                    epochs=150, verbose=1, validation_data=(et_valid_r, y_valid_cat))


# score = model.evaluate([et_test_r, ht_test_r], y_test_cat, verbose=0)
score = model.evaluate(et_test_r, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_history(history, 'loss')
plot_history(history, 'accuracy')


print('############ PREDICT ############')
y_pred = reverse_to_cat(model.predict(et_test_r))
# print(y_pred[:5])
pred_labels = ['Electron' if value == 1
               else 'Jet' for value in y_pred]
print(pred_labels[:10])


print('############ TRUE ############')
y_true = reverse_to_cat(y_test_cat)
# print(y_true[:5])
true_labels = ['Electron' if value == 1 else 'Jet' for value in y_true]
print(true_labels[:10])


for i, v in enumerate(pred_labels[:20]):
    if v == true_labels[i]:
        print('Cazzo Buono')
    else:
        print('NOOOOOOOOOOOOOOO')
