'''Various selection methods that can be used
by the genetic algorithm'''
import random
import numpy as np
np.random.seed(1)


def tournament(population, fitnesses, t_size=3, t_prob=0.7):
    '''Tournament selection. Currently the default.

    Parameters
    ----------
    population : list
        A group of individuals
    fitnesses : list
        Fitness scores corresponding to the given population
    t_size : int
        Number of members participating in one tournament
    t_prob : float
        Probability of the fittest member winning

    Returns
    -------
    parents : list
        Two individuals chosen via two tournaments
    '''
    parents = []
    while len(parents) < 2:
        curr_tournament = []
        t_fitness = []
        while len(curr_tournament) < t_size:
            select = np.random.randint(0, len(population) - 1)
            curr_tournament.append(population[select])
            t_fitness.append(fitnesses[select])
        while len(curr_tournament) >= 1:
            if np.random.random() < t_prob:
                parents.append(curr_tournament[np.argmax(t_fitness)])
                break
            elif len(curr_tournament) == 1:
                parents.append(curr_tournament[0])
                break
            else:
                curr_tournament.remove(curr_tournament[np.argmax(t_fitness)])
                t_fitness.remove(t_fitness[np.argmax(t_fitness)])
    return parents


def rank(population, fitnesses):
    '''Rank selection

    Parameters
    ----------
    population : list
        A group of individuals
    fitnesses : list
        Fitness scores corresponding to the given population

    Returns
    -------
    parents : list
        Two individuals chosen via ranked roulette wheel
    '''
    temp_population = population[:]
    temp_fitnesses = fitnesses[:]
    ranked_population = []
    ranked_fitnesses = []
    ranks = []
    probabilities = []
    curr_rank = 1
    while len(ranked_population) < len(population):
        index = np.argmin(temp_fitnesses)
        ranked_fitnesses.append(min(temp_fitnesses))
        del temp_fitnesses[index]
        ranked_population.append(temp_population[index])
        del temp_population[index]
        ranks.append(curr_rank)
        curr_rank += 1
    for curr_rank in ranks:
        probabilities.append(
            float(curr_rank) / (len(ranks) * (len(ranks) - 1)))
    parents = wheel_parents(ranked_population, probabilities)
    return parents


def roulette(population, fitnesses):
    '''Roulette wheel selection.

    Parameters
    ----------
    population : list
        A group of individuals
    fitnesses : list
        Fitness scores corresponding to the given population

    Returns
    -------
    parents : list
        Two individuals chosen via roulette wheel
    '''
    norm_fitnesses = normalize(fitnesses)
    parents = wheel_parents(population, norm_fitnesses)
    return parents


def wheel_parents(population, probabilities):
    '''Generate roulette wheel according to probabilities
    and select parents

    Parameters
    ----------
    population : list
        A group of individuals
    probabilities : list
        Probability for each individual to be chosen

    Returns
    -------
    parents : list
        Two individuals chosen via a roulette when
        with assigned probabilities

    '''
    wheel = []
    parents = []
    value = 0
    for probability in probabilities:
        value += probability
        wheel.append(value)
    while len(parents) < 2:
        select = random.random()
        for i, slot in enumerate(wheel):
            if select < slot:
                parents.append(population[i])
                break
    return parents


def normalize(fitnesses):
    '''Normalize fitness scores

    Parameters
    ----------
    fitnesses : list
        Fitness scores

    Returns
    -------
    normalized : list
        Normalized fitness scores
    '''
    normalized = []
    total = sum(fitnesses)
    for fitness in fitnesses:
        normalized.append(float(fitness) / total)
    return normalized
