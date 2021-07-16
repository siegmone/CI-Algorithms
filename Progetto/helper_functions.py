import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical


def add_cnn_block(model, input_shape, filter, input_layer=False):
    if input_layer:
        model.add(Conv2D(filters=filter, kernel_size=(1, 1),
                  input_shape=input_shape, padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=filter, kernel_size=(3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
    else:
        model.add(Conv2D(filters=filter, kernel_size=(3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=filter, kernel_size=(3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))


def create_cnn(input_shape, filters, n_dense, hidden_units, dp):
    model = Sequential()
    for (i, f) in enumerate(filters):
        if i == 0:
            add_cnn_block(model=model, input_shape=input_shape,
                          filter=f, input_layer=True)
        else:
            add_cnn_block(model=model, input_shape=input_shape, filter=f)

    model.add(Flatten())

    for _ in range(n_dense):
        model.add(Dense(hidden_units))
        model.add(Activation("relu"))

    model.add(Dropout(dp))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    return model


def create_mlp(input_shape, n_dense, hidden_units, dp):  # MODIFICA MIA ####
    model = Sequential()
    model.add(Flatten())

    for _ in range(n_dense):
        model.add(Dense(hidden_units))
        model.add(Activation("relu"))

    model.add(Dropout(dp))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    return model


def split_data(dataset):
    return dataset['train'], dataset['valid'], dataset['test']


def get_data(data):
    array_data = []
    tvt = split_data(data)
    for dataset in tvt:
        arr = np.zeros(dataset.shape)
        dataset.read_direct(arr)
        array_data.append(arr)
    return array_data


def get_img_data(dataset):
    array_data = get_data(dataset)
    img_data = [arr[:, :, :, 0] for arr in array_data]
    return img_data


def define_inputshape(dataset):
    input_shape_ = (dataset.shape[1], dataset.shape[2], 1)
    return input_shape_


def shape_data(train, valid, test):
    width, height, channels = define_inputshape(train)
    train_r = train.reshape(train.shape[0], width, height, channels)
    valid_r = valid.reshape(valid.shape[0], width, height, channels)
    test_r = test.reshape(test.shape[0], width, height, channels)
    return train_r, valid_r, test_r


def plot_history(model_history, parameter, network):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    ax.plot(model_history.history[f'{parameter}'])
    ax.plot(model_history.history[f'val_{parameter}'])
    ax.set_title(f'model {parameter}')
    ax.set_ylabel(f'{parameter}')
    ax.set_xlabel('epoch')
    if parameter == 'loss':
        ax.set_ylim(bottom=0)
        ax.legend(['train', 'test'], loc='upper right')
    if parameter == 'accuracy':
        ax.set_ylim(top=1)
        ax.legend(['train', 'test'], loc='lower right')
    fig.savefig(f'{network}_{parameter}')


def reverse_to_cat(dataset, content):
    return [np.argmax(y, axis=None, out=None) for y in dataset]


def write_to_file(filepath, content, network):
    with open(filepath, 'a') as f:
        txt = f'''--- {network} Results ---
        Test loss: {content[0]}
        Test accuracy: {content[1]}
        '''
        f.write(txt)


def plot_conf_mat(model, Y_pred, Y_true):
    fig = plt.figure(figsize=(19.20, 10.80))
    cm = confusion_matrix(Y_pred, Y_true)
    sn.heatmap(cm, annot=True, annot_kws={"size": 14})
    fig.savefig('MLP confusion matrix')
