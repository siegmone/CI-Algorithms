import keras
import numpy
import pandas as pd
import pygad
import pygad.cnn
import pygad.gacnn
import pygad.kerasga
import tensorflow
import tensorflow.keras
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential


def fitness_func(solution, sol_idx):
    global GACNN_instance, data_inputs, data_outputs

    predictions = GACNN_instance.population_networks[sol_idx].predict(
        data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions / data_outputs.size) * 100

    return solution_fitness


def callback_generation(ga_instance):
    global GACNN_instance, last_fitness

    population_matrices = pygad.gacnn.population_as_matrices(population_networks=GACNN_instance.population_networks,
                                                             population_vectors=ga_instance.population)

    GACNN_instance.update_population_trained_weights(
        population_trained_weights=population_matrices)

    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solutions_fitness))


df = pd.read_csv('sensor_readings_4.data', names=[
                 'd', 'r', 't', 'y', 'Direction'], header=None)
print(df)
dizionario = {'Move-Forward': 0, 'Sharp-Right-Turn': 1,
              'Slight-Right-Turn': 2, 'Slight-Left-Turn': 3}
df["Direction"].replace(dizionario, inplace=True)
dataset = tensorflow.data.Dataset.from_tensor_slices(
    (df.values, df.pop('Direction').values))

print(list(dataset.as_numpy_iterator()))


sample_shape = data.shape[1:]
num_classes = 4

data_inputs = data
data_outputs = labels

input_layer = pygad.cnn.Input1D(input_shape=sample_shape)
conv_layer1 = pygad.cnn.Conv1D(num_filters=2,
                               kernel_size=3,
                               previous_layer=input_layer,
                               activation_function="relu")
average_pooling_layer = pygad.cnn.AveragePooling1D(pool_size=5,
                                                   previous_layer=conv_layer1,
                                                   stride=3)

flatten_layer = pygad.cnn.Flatten(previous_layer=average_pooling_layer)
dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes,
                               previous_layer=flatten_layer,
                               activation_function="softmax")

model = pygad.cnn.Model(last_layer=dense_layer2,
                        epochs=1,
                        learning_rate=0.01)

model.summary()


GACNN_instance = pygad.gacnn.GACNN(model=model,
                                   num_solutions=4)

# GACNN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

# population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
# If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
population_vectors = pygad.gacnn.population_as_vectors(
    population_networks=GACNN_instance.population_networks)

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
initial_population = population_vectors.copy()

# Number of solutions to be selected as parents in the mating pool.
num_parents_mating = 2

num_generations = 10  # Number of generations.

# Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
mutation_percent_genes = 0.1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

# Predicting the outputs of the data using the best solution.
predictions = GACNN_instance.population_networks[solution_idx].predict(
    data_inputs=data_inputs)
print("Predictions of the trained network : {predictions}".format(
    predictions=predictions))

# Calculating some statistics
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct / data_outputs.size)
print("Number of correct classifications : {num_correct}.".format(
    num_correct=num_correct))
print("Number of wrong classifications : {num_wrong}.".format(
    num_wrong=num_wrong.size))
print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))
