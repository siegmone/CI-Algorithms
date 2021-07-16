import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from helper_functions import *
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adadelta, Adam, Nadam
from tensorflow.keras.utils import to_categorical

plt.style.use('seaborn-whitegrid')

open('CNN_Results.txt', 'w').close()
open('MLP_Results.txt', 'w').close()


def run_neural_network(model, train, valid, test, network):
    opt = Adam(epsilon=0.01, learning_rate=0.00001)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt, metrics=['accuracy'])

    history = model.fit(x=train[0], y=train[1], batch_size=128,
                        epochs=150, verbose=1, validation_data=(valid[0], valid[1]))

    score = model.evaluate(test[0], test[1], verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    write_to_file(filepath=f'{network}_Results.txt',
                  content=score, network=network)

    plot_history(history, 'loss', network=network)
    plot_history(history, 'accuracy', network=network)

    print('============ PREDICT ============')
    y_pred = reverse_to_cat(model.predict(train[0]))
    pred_labels = ['Electron' if value == 1 else 'Jet' for value in y_pred]
    print(pred_labels[:20])

    print('============ TRUE ============')
    y_true = reverse_to_cat(train[1])
    true_labels = ['Electron' if value == 1 else 'Jet' for value in y_true]
    print(true_labels[:20])

    plot_conf_mat(model=model, Y_pred=y_pred, Y_true=y_true)

    return score


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

hl_m_train = np.hstack((hl_train, mass_train))
hl_m_valid = np.hstack((hl_valid, mass_valid))
hl_m_test = np.hstack((hl_test, mass_test))

y_train_cat = to_categorical(y_train)
y_valid_cat = to_categorical(y_valid)
y_test_cat = to_categorical(y_test)


mlp_hl = create_mlp(input_shape=(8, 1), n_dense=3, hidden_units=128, dp=0.1)
cnn_et = create_cnn(input_shape=et_input_shape, filters=(
    16, 32, 64, 128), n_dense=3, hidden_units=128, dp=0.1)


# Running cnn on et images
print('######################## RUNNING CNN_ET ########################')
run_neural_network(cnn_et, (et_train_r, y_train_cat),
                   (et_valid_r, y_valid_cat), (et_test_r, y_test_cat), 'CNN')

# Running mlp on hl and mass
print('######################## RUNNING MLP_HL (10 times) ########################')
scores = []
for r in range(10):
    print(
        f'-------------------- Running {r} iteration of MLP --------------------')
    score_mlp = run_neural_network(
        mlp_hl, (hl_m_train, y_train_cat), (hl_m_valid, y_valid_cat), (hl_m_test, y_test_cat), 'MLP')

    scores.append(score_mlp)

scores = np.array(scores)
mean_loss, mean_accuracy = np.mean(scores.T[0]), np.mean(scores.T[1])
print('Mean MLP loss', mean_loss)
print('Mean MLP accuracy', mean_accuracy)
