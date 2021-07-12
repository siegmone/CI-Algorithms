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


def to_cat(dataset):
    arr = np.zeros(dataset.shape)
    dataset.read_direct(arr)
    return to_categorical(arr)


y_train_cat, y_valid_cat, y_test_cat = to_cat(
    y_train), to_cat(y_valid), to_cat(y_test)

print(y_train_cat.shape)


def create_cnn(width, height, depth, filters, n_dense, hidden_units, dp):
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
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(f, (4, 4), padding="same")(x)
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
        x = Dense(32)(x)
        x = Activation("relu")(x)
        # construct the CNN
        model = Model(inputs, x)
        # return the CNN
        return model


# cnn_ht = create_cnn(width=32, height=32, depth=1, filters=(
#     8, 16, 32), n_dense=2, hidden_units=80, dp=0.5)
# cnn_et = create_cnn(width=31, height=31, depth=1, filters=(
#     32, 64, 128), n_dense=2, hidden_units=160, dp=0.0)


# def create_cnn(width, height, depth, filters, n_dense, hidden_units, dp):
#     inputShape = (height, width, depth)
#     model = Sequential()
#     model.add(Conv2D(filters=128, kernel_size=(3, 3),
#               activation='relu', input_shape=inputShape))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(64, activation='sigmoid'))
#     model.add(Dense(4, activation='softmax'))
#
#     return model


cnn_ht = create_cnn(width=32, height=32, depth=1, filters=(
    8, 16, 32), n_dense=2, hidden_units=90, dp=0.4)
cnn_et = create_cnn(width=31, height=31, depth=1, filters=(
    32, 64, 128), n_dense=2, hidden_units=180, dp=0.1)


# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([cnn_ht.output, cnn_et.output])
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(2, activation="softmax")(combinedInput)
#x = Dense(1, activation="linear")(x)
# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[cnn_ht.input, cnn_et.input], outputs=x)

opt = Nadam(0.0001, lr=0.0001)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy'])

# train the model

print("[INFO] training model...")

model.fit(

    x=[get_images(ht_train), get_images(et_train)], y=y_train_cat,

    validation_data=(
        [get_images(ht_valid), get_images(et_valid)], y_valid_cat),

    epochs=25, batch_size=128, verbose=1)

score = model.evaluate(
    [get_images(ht_test), get_images(et_test)], y_test_cat, verbose=1)
# make predictions on the testing data

print("[INFO] predicting electrons...")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
