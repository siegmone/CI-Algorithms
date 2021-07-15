import os

# import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam
from tensorflow.keras.utils import to_categorical

data = h5py.File('data.h5', 'r')

# split data func


def split_data(dataset):
    return dataset['train'], dataset['valid'], dataset['test']


def get_images(dataset):
    return dataset[:, :, :, 0]


et, hl, ht, mass, y = data['et'], data['hl'], data['ht'], data['mass'], data['y']
et_train, et_valid, et_test = split_data(et)
hl_train, hl_valid, hl_test = split_data(hl)
ht_train, ht_valid, ht_test = split_data(ht)
mass_train, mass_valid, mass_test = split_data(mass)
y_train, y_valid, y_test = split_data(y)


def to_array(dataset):
    arr = np.zeros(dataset.shape)
    dataset.read_direct(arr)
    return arr


et_train_r, et_valid_r, et_test_r = to_array(
    et_train), to_array(et_valid), to_array(et_test)
ht_train_r, ht_valid_r, ht_test_r = to_array(
    ht_train), to_array(ht_valid), to_array(ht_test)


tot_train_examples, tot_valid_examples, tot_test_examples, width, height, depth = et_train_r.shape[0], et_valid_r.shape[z0] et_test_r.shape[z0]
et_train_r = et_train_r.reshape(tot_train_examples, width, height, depth)
et_valid_r = et_valid_r.reshape(tot_train_examples, width, height, depth)
et_test_r = et_test_r.reshape(tot_train_examples, width, height, depth)
ht_train_r = ht_train_r.reshape(tot_train_examples, width, height, depth)
ht_valid_r = ht_valid_r.reshape(tot_train_examples, width, height, depth)
ht_test_r = ht_test_r.reshape(tot_train_examples, width, height, depth)


y_train_r, y_valid_r, y_test_r = to_array(
    y_train), to_array(y_valid), to_array(y_test)

y_train_cat, y_valid_cat, y_test_cat = to_categorical(
    y_train_r), to_categorical(y_valid_r), to_categorical(y_test_r)

y_train_cat, y_valid_cat, y_test_cat = y_train_cat.reshape(y_train_cat.shape + (
    1,)), y_valid_cat.reshape(y_valid_cat.shape + (1,)), y_test_cat.reshape(y_test_cat.shape + (1,))


hl_train_cat, hl_valid_cat, hl_test_cat = to_array(
    hl_train), to_array(hl_valid), to_array(hl_test)


output = 32
#### CNN IMMAGINI ####


def create_cnn(width, height, depth, filters, n_dense, hidden_units, dp):
    global output
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(f, (1, 1), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    for _ in range(n_dense):
        x = Dense(hidden_units)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(dp)(x)
        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(output)(x)
        x = Activation("relu")(x)
        # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model


cnn_ht = create_cnn(width=32, height=32, depth=1, filters=(
    47, 47, 47), n_dense=2, hidden_units=146, dp=0.0)
cnn_et = create_cnn(width=31, height=31, depth=1, filters=(
    47, 47, 47), n_dense=2, hidden_units=146, dp=0.0)

x = Dense(output, activation="relu")(cnn_et.output)
x = Flatten()(x)
x = Dense(2, activation="softmax")(x)

model = Model(inputs=[cnn_ht.input, cnn_et.input], outputs=x)


#### MLP ####
# Creazione del Modello
# hl_train_cat = hl_train_cat.reshape(hl_train_cat.shape)
# hl_test_cat = hl_test_cat.reshape(hl_test_cat.shape)


# input_shape = (7,)
# model = Sequential()
# model.add(Dense(128, activation='sigmoid', input_shape=input_shape))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='softmax'))
# model.add(Dropout(0.2))
# model.build(input_shape)
# model.summary()

opt = Adam(0.0001)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy'])

# train the model

print("[INFO] training model...")

model.fit(x=[ht_train_r, et_train_r], y=y_train_cat,

          validation_data=([ht_valid_r, et_valid_r], y_valid_cat),

          epochs=25, batch_size=128, verbose=1)

score = model.evaluate([ht_test_r, et_test_r], y_valid_cat, verbose=1)


# model.fit(
#
#     x=get_images(et_train), y=y_train_cat,
#
#     validation_data=(get_images(et_valid), y_valid_cat),
#
#     epochs=25, batch_size=256, verbose=1)

# model.fit(
#
#     x=hl_train_cat, y=y_train_cat,
#
#     validation_data=(hl_valid_cat, y_valid_cat),
#
#     epochs=25, batch_size=128, verbose=1)
#
# score = model.evaluate(hl_test_cat, y_test_cat, verbose=1)


# make predictions on the testing data

print("[INFO] predicting electrons...")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
