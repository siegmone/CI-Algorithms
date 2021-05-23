import random
import time
import matplotlib.pyplot as plt
import numpy as np

from deap import base, creator, tools
from operator import mul
from math import cos, sqrt
from functools import reduce


plt.style.use('seaborn-whitegrid')



# ================================================
#         DIFFERENTIAL EVOLUTION ALGORITHM
# ================================================
# io amo simone

NDIM = 2  # Problem Dimension

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def evaluate(x): # Griewank
    """Evaluates Individual's Fitness"""
    s = sum(map(lambda h: (h**2) / 4000, x))
    c = [cos(x[i - 1]) / sqrt(i) for i in range(1, len(x) + 1)]
    p = reduce(mul, c, 1)
    return (1 + s - p,)


toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -20, 20)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, n=NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", evaluate)

# Parameters
CR = 0.3
F = 1
pop_size = 25000
NGEN = 100
f = 4

pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


def de(toolbox, pop, CR, F, NGEN, stats, hof, plot=False):

    fig, ax = plt.subplots(figsize=(20, 10))
    # ax.set_ylim(bottom=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals
    fitnesses = map(toolbox.evaluate, pop)
    fits = []
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        fits.append(ind.fitness.values[0])

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    print(logbook.stream)

    gens = []
    avgs = []

    for g in range(1, NGEN + 1):
        # figure, axs = plt.subplots(figsize=(20, 10))
        # axs.set_ylim(0, max(fits))
        # fl = map(abs, [x[0] for x in pop])
        # ab = max(list(fl))
        # axs.set_xlim(-ab, ab)

        gens.append(g)
        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])
            y.fitness.values = toolbox.evaluate(y)
            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        # print(logbook.stream)
        avgs.append(record["avg"])

        print(pop[0])

        fits = [ind.fitness.values[0] for ind in pop]

        # axs.scatter(pop, fits, marker='o')
        # figure.savefig(f"try/{g}")
        # plt.close(figure)
        # del axs
        if record["min"] == 0:
            break

    print("Best individual is",
          hof[0], "With fitness value:", hof[0].fitness.values[0])
    if plot == True:
        ax.plot(gens, avgs, marker="o")
        fig.savefig("diff.png")






# =============================================
#         SIMPLE EVOLUTIONARY ALGORITHM
# =============================================



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return individual


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def eaSimple():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return pop