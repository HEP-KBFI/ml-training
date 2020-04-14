'''Main functions for the genetic algorithm'''
import inspect
import numpy as np
from machineLearning.machineLearning import universal as ut
from machineLearning.machineLearning import ga_selection as select
from machineLearning.machineLearning import ga_crossover as gc


class Individual:
    ''' A class used to represent an individual member of a population

    Attributes
    ----------
    values : dict
        Set of parameter values
    subpop : int
        Number denoting the subpopulation to which the individual
        belongs
    fitness : float
        Fitness score of the individual

    Methods
    -------
    add_result(score, pred_train, pred_test, feature_importance,
    fitness)
        Adds the the evaluation results as attributes to the individual
    merge()
        Assignes the subpopulation number to be 0
    '''
    def __init__(self, values, subpop):
        self.values = values
        self.subpop = subpop
        self.fitness = None

    def __eq__(self, other):
        return self.values == other.values

    def add_result(
            self,
            fitness,
            score=None,
        ):
        '''Adds the the evaluation results as attributes to the
        individual'''
        self.fitness = fitness

    def merge(self):
        '''Assignes the subpopulation number to be 0'''
        self.subpop = 0


# OK!
def assign_individuals(population, subpop):
    '''Assigns generated values to members of the class Individual

    Parameters
    ----------
    population : list
        List of generated values for the population
    subpop : int
        Number denoting the current subpopulation

    Returns
    -------
    individuals : list
        The population as a list of individuals
    '''
    individuals = []
    for member in population:
        individual = Individual(member, subpop)
        individuals.append(individual)
    return individuals


# OK!
def create_population(settings, parameters, create_set):
    '''Creates a randomly generated population

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population

    Returns
    -------
    population : list
        Randomly generated population
    '''
    population = []
    size = settings['sample_size']
    num = settings['sub_pops']
    if num > size:
        return population
    for i in range(num):
        if i == 0:
            sub_size = size//num + size % num
        else:
            sub_size = size//num
        sub_population = create_set(parameters, sub_size)
        sub_population = assign_individuals(sub_population, i)
        population += sub_population
    return population


# OK!
def separate_subpopulations(population, settings):
    '''Separate the population into subpopulations

    Parameters
    ----------
    population : list
        The entire population
    settings : dict
        Settings of the genetic algorithm

    Returns
    -------
    separated : list
        List of separated subpopulations
    '''
    subpopulations = []
    separated = []
    for i in range(settings['sub_pops']):
        subpopulations.append([])
    for member in population:
        index = member.subpop
        subpopulations[index].append(member)
    for subpopulation in subpopulations:
        if subpopulation:
            separated.append(subpopulation)
    return separated


# OK!
def unite_subpopulations(subpopulations):
    '''Reunite separated subpopulations

    Parameters
    ----------
    subpopulations : list
        List of subpopulations

    Returns
    -------
    population : list
        The entire population
    '''
    population = []
    for subpopulation in subpopulations:
        population += subpopulation
    return population


def set_num(amount, population):
    '''Set num as the amount indicated for the given population.
    If given a float between 0 and 1, num is set
    as a given fraction of the population.
    If given a int larger than 1, num is set as that int.
    If given any other number, num is set as 0.

    Parameters
    ----------
    amount : float or int
        Given number
    population : list
        Current population

    Returns
    -------
    num : int
        Number of members of the population indicated
        by the amount given.
    '''
    # Given is a number
    if amount >= 1 and isinstance(amount, int):
        num = amount
    # Given is a fraction
    elif 0 < amount < 1:
        num = int(round(len(population) * amount))
    # Given an invalid number
    else:
        num = 0
    return num


# OK!
def fitness_list(population):
    '''Generate a list of fitness scores for the given population

    Parameters
    ----------
    population : list
        The given population

    Returns
    -------
    fitnesses : list
        List of fitness scores for the given population
    '''
    fitnesses = []
    for member in population:
        fitnesses.append(member.fitness)
    return fitnesses


def population_list(population):
    '''Convert the population from a list of individuals to a list of
    values

    Parameters
    ----------
    population : list
        The given population

    Returns
    value_list : list
        The given population as a list of values
    '''
    value_list = []
    for member in population:
        value_list.append(member.values)
    return value_list


def fitness_calculation(population, settings, evaluate):
    '''Calculate the fitness scores of the given generation

    Parameters
    ----------
    population : list
        A group of individuals to be evaluated
    settings : dict
        Settings of the genetic algorithm
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    population : list
        A group of individuals
    '''
    eval_pop, rest_pop = arrange_population(population)
    if eval_pop:
        fitnesses = evaluate(population_list(eval_pop), settings)
        for i, member in enumerate(eval_pop):
            member.add_result(fitnesses[i])
        population = eval_pop + rest_pop
    return population

# OK!
def elitism(population, settings):
    '''Preserve best performing members of the previous generation

    Parameters
    ----------
    population : list
        A group of individuals
    settings : dict
        Settings of the genetic algorithm

    Returns
    -------
    elite : list
        Best performing members of the given population
    '''
    num = set_num(settings['elites'], population)
    population_copy = population.copy()
    fitnesses = fitness_list(population_copy)
    elite = []
    while num > 0:
        index = np.argmax(fitnesses)
        del fitnesses[index]
        elite.append(population_copy.pop(index))
        num -= 1
    return elite

# OK!
def culling(
        population,
        settings,
        parameters,
        create_set,
        evaluate,
        subpop=0
):
    '''Cull worst performing members
    and replace them with random new ones

    Parameters
    ----------
    population : list
        Population to be culled
    settings : dict
        Settings of the genetic algorithm
    parameters : dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores
    subpop : int (optional)
        Which subpopulation is being culled

    Returns
    -------
    population : list
        New population with worst performing members replaced
    '''
    num = set_num(settings['culling'], population)
    fitnesses = fitness_list(population)
    size = num
    if num == 0:
        return population
    while num > 0:
        index = np.argmin(fitnesses)
        del fitnesses[index]
        del population[index]
        num -= 1
    new_members = create_set(parameters, size)
    new_members = assign_individuals(new_members, subpop)
    new_members = fitness_calculation(new_members, settings, evaluate)
    population += new_members
    return population


# OK!
def new_population(
        population,
        settings,
        parameters,
        create_set,
        evaluate,
        subpop=0
):
    '''Create the next generation population.

    Parameters
    ----------
    population : list
        Current set of individuals
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores
    subpop : int (optional)
        Which subpopulation is being updated

    Returns
    -------
    next_population : list
        Newly generated set of individuals
    '''
    population = culling(
        population, settings, parameters, create_set, evaluate, subpop)
    offsprings = []
    next_population = elitism(population, settings)
    fitnesses = fitness_list(population)
    while len(offsprings) < (len(population) - len(next_population)):
        parents = select.tournament(population_list(population), fitnesses)
        offspring = gc.uniform_crossover(
            parents, parameters, settings['mut_chance'])
        if offspring not in next_population:
            offsprings.append(offspring)
    next_population += assign_individuals(offsprings, subpop)
    return next_population


def arrange_population(population):
    '''Arrange population into separate lists in preparation for
    evaluation

    Parameters
    ----------
    population : list
        Population to be arranged

    Returns
    -------
    eval_pop : list
        Members from all subpopulations to be evaluated
    rest_pop : list
        Members from all subpopulations already evaluated
    '''
    eval_pop = []
    rest_pop = []
    for member in population:
        # Find members not yet evaluated
        if member.fitness is None:
            eval_pop.append(member)
        # Find member already evaluated
        else:
            rest_pop.append(member)
    return eval_pop, rest_pop


# OK!
def merge_subpopulations(subpopulations):
    '''Merge subpopulations into one population

    Parameters
    ----------
    subpopulations : list
        List of subpopulations to merge

    Returns
    -------
    population : list
        A merged population
    '''
    population = unite_subpopulations(subpopulations)
    for member in population:
        member.merge()
    return population


def finish_subpopulation(
        subpopulations,
        finished_subpopulations,
        improvements,
        threshold
):
    '''Separate out a subpopulation that has reached the improvement threshold

    Parameters
    ----------
    subpopulations : list
        List of all current subpopulations
    finished_subpopulations : list
        List of already finished subpopulations
    improvements : list
        Improvement scores for the current subpopulations
    threshold : float
        Threshold value for the improvement score

    Returns
    -------
    finished_subpopulations : list
        Updated list of finished subpopulations
    remaining_subpopulations : list
        Subpopulations to continue evolving
    '''
    remaining_subpopulations = []
    for i, improvement in enumerate(improvements):
        if improvement <= threshold:
            finished_subpopulations.append(subpopulations[i])
        else:
            remaining_subpopulations.append(subpopulations[i])
    return finished_subpopulations, remaining_subpopulations


def evolve(
        population,
        settings,
        parameters,
        create_set,
        evaluate
):
    '''Evolve a population until reaching the threshold
    or maximum number of iterations. In case of subpopulations, first
    evolve all subpopulations until reaching either criteria, then
    evolve the merged population until reaching either criteria

    Parameters
    ----------
    population : list
        Initial population
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    best_parameters : dict
        best parameters found for the current problem.
    '''
    if settings['sub_pops'] > 1:
        population = evolve_subpopulations(population, settings, create_set)
    iteration = 0
    improvement = 1
    improvements = []
    avg_scores = []
    while (iteration <= settings['iterations']
           and improvement > settings['threshold']):
        if iteration != 0:
            print('::::: Iteration: ' + str(iteration) + ' :::::')
            population = new_population(
                population, settings, parameters, create_set, evaluate)
        population = fitness_calculation(
            population, settings, evaluate)
        fitnesses = fitness_list(population)
        avg_scores.append(np.mean(fitnesses))
        improvements, improvement = ut.calculate_improvement(
            avg_scores,
            improvements,
            settings['threshold']
        )
        iteration += 1
    index_of_best = np.argmax(fitness_list(population))
    best_parameters = population[index_of_best]
    return best_parameters


def evolve_subpopulations(
        population,
        settings,
        parameters,
        create_set,
        evaluate
):
    '''Evolve subpopulations until reaching the threshold
    or maximum number of iterations. In case of subpopulations, first
    evolve all subpopulations until reaching either criteria, then
    evolve the merged population until reaching either criteria

    Parameters
    ----------
    population : list
        Initial population
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    best_parameters : dict
        best parameters found for the current problem.
    '''
    iteration = 0
    finished_subpopulations = []
    curr_improvement = []
    improvements = {}
    avg_scores = {}
    while (iteration <= settings['iterations'] and population):
        if iteration != 0:
            print('::::: Iteration:' + str(iteration) + ' :::::')
            new_subpopulations = []
            for subpopulation in subpopulations:
                new_subpopulation = new_population(
                    subpopulation,
                    settings,
                    parameters,
                    create_set,
                    evaluate,
                    subpopulation[0].subpop
                )
                new_subpopulations.append(new_subpopulation)
            population = unite_subpopulations(new_subpopulations)
        population = fitness_calculation(
            population, settings, evaluate)
        subpopulations = separate_subpopulations(population, settings)
        curr_improvements = []
        for subpopulation in subpopulations:
            index = subpopulation[0].subpop
            fitnesses = fitness_list(subpopulation)
            if iteration == 0:
                avg_scores[index] = []
                improvements[index] = []
            avg_scores[index].append(np.mean(fitnesses))
            improvements[index], curr_improvement = ut.calculate_improvement(
                avg_scores[index],
                improvements[index],
                settings['threshold']
            )
            curr_improvements.append(curr_improvement)
        finished_subpopulations, subpopulations = finish_subpopulation(
            subpopulations, finished_subpopulations,
            curr_improvements, settings['threshold']
        )
        population = unite_subpopulations(subpopulations)
        iteration += 1
    print('::::: Merging subpopulations :::::')
    subpopulations += finished_subpopulations
    population = merge_subpopulations(subpopulations)
    return population


def evolution(settings, parameters, create_set, evaluate):
    '''Evolution of the parameter values for hyperparameter optimization

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    result : dict
        Result of the run of the genetic algorithm
    '''
    print('\n::::: Generating initial population :::::')
    population = create_population(settings, parameters, create_set)
    print('\n::::: Starting evoliton :::::')
    best_parameters = evolve(
        population, settings, parameters, create_set, evaluate)
    print('\n::::: Evolved! :::::')
    return best_parameters
