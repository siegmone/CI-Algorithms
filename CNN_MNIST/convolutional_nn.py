import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

plt.style.use('seaborn-whitegrid')
# tf.config.optimizer.set_jit(True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

tot_train_examples = x_train.shape[0]
tot_test_examples = x_test.shape[0]
width = 28
height = 28
channels = 1
inp_s = (width, height, channels)
num_classes = 10
bs = 32
epochs = 20

x_train_r = x_train.reshape(tot_train_examples, width, height, channels)
x_test_r = x_test.reshape(tot_test_examples, width, height, channels)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


def add_2DConv_Layer(model, filters, kernel_size, input_shape, activation_f):
    model.add(Conv2D(filters=filters, kernel_size=kernel_size,
                     activation=activation_f, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))


#### CNN ####
# Creazione del Modello
activation = "relu"
model = Sequential()
add_2DConv_Layer(model=model, filters=16, kernel_size=(
    1, 1), input_shape=inp_s, activation_f=activation)
add_2DConv_Layer(model=model, filters=32, kernel_size=(
    3, 3), input_shape=inp_s, activation_f=activation)
add_2DConv_Layer(model=model, filters=64, kernel_size=(
    3, 3), input_shape=inp_s, activation_f=activation)
add_2DConv_Layer(model=model, filters=128, kernel_size=(
    1, 1), input_shape=inp_s, activation_f=activation)

model.add(Flatten())
model.add(Dense(num_classes, activation="softmax"))

opt = tf.keras.optimizers.Adam(epsilon=0.001)

# Compilazione, Addestramento e Validazione del Modello
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train_r, y_train_cat, batch_size=bs, epochs=epochs,
                    verbose=1, validation_data=(x_test_r, y_test_cat))
score = model.evaluate(x_test_r, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(f'trained_model')

# summarize history for accuracy
plt.ylim()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(
    f'Graphics\Accuracy_4xConvLayers_16-32-64-128-Filters_{bs}-batch-size_{activation}_Adam')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(
    f'Graphics\Loss_4xConvLayers_16-32-64-128-Filters_{bs}-batch-size_{activation}_Adam')
plt.show()
