import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import keras
import pandas as pd
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import seaborn as sn
from sklearn import metrics

plt.style.use('seaborn-whitegrid')



def reverse_categorical(Y):
    return [np.argmax(y, axis=None, out=None) for y in Y]



def split(set_dati, percentuale):
    flag = int(len(set_dati)*percentuale/100)
    X_train = set_dati[0:flag]
    X_test = set_dati[flag:len(set_dati)]
    return X_train, X_test

def lettura_dataset(num_sensori):
    df = pd.read_csv(dati_sensori, header=None)
    df = df.sample(frac=1).reset_index(drop = True)
    dizionario = {'Move-Forward':0, 'Sharp-Right-Turn':1, 'Slight-Right-Turn':2, 'Slight-Left-Turn':3}
    df[num_sensori].replace(dizionario, inplace=True)
    risultati = {}
    y = df.iloc[:, num_sensori].values
    y_train, y_test = split(y, 90)
    X = df.iloc[:, :num_sensori].values
    x_train, x_test = split(X, 90)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    return x_train, x_test, y_train_cat, y_test_cat


def create_MLP(num_sensori):
    print('------- MLP con ' + str(num_sensori) + ' sensori ---------')
        #### MLP ####
    # Creazione del Modello
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def modello():
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt, metrics=['accuracy'])
    history = model.fit(x_train, y_train_cat, batch_size=bs, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test_cat))
    score = model.evaluate(x_test, y_test_cat, verbose=1)
    print('------- Predict --------')
    dizionario = {0:'Move-Forward',1 :'Sharp-Right-Turn', 2:'Slight-Right-Turn', 3:'Slight-Left-Turn'}
    y_pred = reverse_categorical(model.predict(x_test))
    y_pred_comm = [dizionario[value] for value in y_pred]
    print(y_pred_comm[0:10])
    print('-------- Target --------')
    y_targ = reverse_categorical(y_test_cat)
    y_targ_comm = [dizionario[value] for value in y_targ]
    print(y_targ_comm[0:10])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history, score

def plot_accuracy(history, num_sensori):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label = ' Training')
    ax.plot(history.history['val_accuracy'], label = 'Validation')
    ax.set_title('MLP accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(loc='lower right')
    fig.savefig('MLP Accuracy ' + str(num_sensori))

def plot_loss(history, num_sensori):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label = 'Training')
    ax.plot(history.history['val_loss'], label = 'Validation')
    ax.set_title('MLP loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    fig.savefig('MLP Loss ' + str(num_sensori))

def conf_mat(X, Y):
    rounded_labels=np.argmax(Y, axis=1)
    y_pred=model.predict(X)
    rounded_predictions=np.argmax(y_pred, axis=1)
    cm = metrics.confusion_matrix(rounded_labels, rounded_predictions)
    fig = plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    fig.savefig('MLP confusion matrix ' + str(num_sensori))


def bar_chart(X):
    plt.clf()
    scores = np.array(X)
    scores.shape = (3,2)
    sensors = ('2', '4', '24')
    y_sensors = np.array((1,2,3))
    print(y_sensors.shape)
    y_sensors.shape = (3,1)
    y = np.hstack((y_sensors, scores))
    df = pd.DataFrame(y, columns=["# Sensors", "Loss", "Accuracy"] )
    df.plot(x="# Sensors", y=["Accuracy", "Loss"],  kind="bar" )
    plt.xticks(range(0,3), sensors, rotation = 'horizontal')
    plt.savefig('Bar chart')



###### Selezione dataset #######
# num_sensori = 24
# dati_sensori = 'sensor_readings_' + str(num_sensori) + '.data'
# bs = 8
# epochs = 150
# input_shape = (1,num_sensori)
# num_classes = 4
# x_train, x_test, y_train_cat, y_test_cat = lettura_dataset(num_sensori)
# model = create_MLP(num_sensori)
# history, score = modello()
# plot_accuracy()
# plot_loss()
# conf_mat()


###### Ciclo for su tutti i dataset ######
data = []
for i in [2, 4, 24]:
    scores = []
    dati_sensori = 'sensor_readings_' + str(i) + '.data'
    num_sensori = i
    bs = 32
    epochs = 150
    input_shape = (1, num_sensori)
    num_classes = 4
    x_train, x_test, y_train_cat, y_test_cat = lettura_dataset(num_sensori)

    for j in range(50):
        print('-------------------- Run ' + str(j + 1) + ' --------------------')
        model = create_MLP(num_sensori)
        history, score = modello()
        scores.append(score)
        plot_accuracy(history, num_sensori)
        plot_loss(history, num_sensori)
        conf_mat(x_test, y_test_cat)
    print(scores)
    scores = np.array(scores)
    mean_loss, mean_accuracy = np.mean(scores.T[0]), np.mean(scores.T[1])
    data.append([mean_loss, mean_accuracy])
    print('Mean loss', np.mean(scores.T[0]))
    print('Mean accuracy', np.mean(scores.T[1]))

print(data)

bar_chart(data)
