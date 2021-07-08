import keras
import numpy
import pandas as pd
import pygad
import pygad.kerasga
import tensorflow
import tensorflow.keras
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential


def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    mae = tensorflow.keras.losses.MeanAbsoluteError()
    abs_error = mae(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error

    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))


df = pd.read_csv('sensor_readings_4.data', names=[
                 'd', 'r', 't', 'y', 'Direction'], header=None)
print(df)
dizionario = {'Move-Forward': 0, 'Sharp-Right-Turn': 1,
              'Slight-Right-Turn': 2, 'Slight-Left-Turn': 3}
df["Direction"].replace(dizionario, inplace=True)
dataset = tensorflow.data.Dataset.from_tensor_slices(
    (df.values, df.pop('Direction').values))


for data, labels in dataset.take(1):
    print(data)
    print(labels)

#print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
#print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))
# print(x_train)
num_classes = 4
bs = 1
epochs = 100


input_layer = tensorflow.keras.layers.Input(1)
dense_layer1 = tensorflow.keras.layers.Dense(1, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(
    1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

# Data inputs
data_inputs = data
print(data)
# Data outputs
data_outputs = labels

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 50  # Number of generations.
# Number of solutions to be selected as parents in the mating pool.
num_parents_mating = 5
# Initial population of network weights
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(
    title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))


predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)
