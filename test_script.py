import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from multiprocessing import Pool, cpu_count
from algos import eaSimple_cp

def _create():
    creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FeatureSelect)
    return creator


def _init_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox
        

def _default_toolbox(indpb=0.1, tournsize=3):
    toolbox = _init_toolbox()
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("evaluate", eval_ind)
    return toolbox

def _stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def _history():
    history = tools.History()
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)
    return history

def eval_ind(individual, n_splits=5):
    np_ind = np.asarray(individual)
    if np.sum(np_ind) == 0:
        fitness = 0.0
    else:
        fitness = cross_val_score(model, x_train[:, np_ind == 1], y_train, cv=n_splits, verbose=1).mean()
    
    print(f"Individual: {individual}  Fitness_score: {fitness}")
        
    return fitness,


# main 
CXPB = 0.5
MUTPB = 0.2
NGEN = 3
n_pop = 4
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
x_train, x_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2)
model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
n_features = x_train.shape[1]

creator = _create()
toolbox = _default_toolbox()
pop = toolbox.population(n_pop)
history = _history()
history.update(pop)

hof = tools.HallOfFame(5)
stats = _stats()

pool = Pool(processes=cpu_count())
toolbox.register("map", pool.map)

pop, log = eaSimple_cp(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                                stats=stats, halloffame=hof, verbose=True)

best_ind = hof.items

rand = np.random.choice(X_digits.shape[1], size=sum(best_ind[0]), replace=False)
cross_val_score(model, X_digits[:, rand], y_digits, scoring="balanced_accuracy", cv=5).mean()
