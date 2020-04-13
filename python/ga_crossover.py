'''Functions used by the genetic algorithm for crossover
and mutation'''
import random


# CROSSOVER FUNCTIONS
def kpoint_crossover(parents, parameters, mutation_chance=0, k=1):
    '''Crossover of parents via k-point crossover.
    Default is single-point crossover.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit
    k : int
        number of points in k-point crossover

    Returns
    -------
    offspring : dict
        New individual
    '''
    points = []
    offspring = {}
    parent1 = encode_parent(parents[0], parameters)
    parent2 = encode_parent(parents[1], parameters)
    for i in range(k):
        point = int(round(random.random() * 31))
        points.append(point)
    for key in parent1:
        parent1_chromosome = parent1[key]
        parent2_chromosome = parent2[key]
        curr_chromosome = parent1_chromosome
        offspring_chromosome = ''
        for i in range(len(parent1_chromosome)):
            offspring_chromosome += curr_chromosome[i]
            if i in points:
                if curr_chromosome == parent1_chromosome:
                    curr_chromosome = parent2_chromosome
                else:
                    curr_chromosome = parent1_chromosome
        if mutation_chance != 0:
            offspring_chromosome = chromosome_mutate(
                offspring_chromosome, mutation_chance)
        offspring[key] = offspring_chromosome
    offspring = decode_offspring(offspring, parameters)
    offspring = mutation_fix(offspring, parameters)
    return offspring


def uniform_crossover(parents, parameters, mutation_chance=0):
    '''Crossover of parents via uniform crossover.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit

    Returns
    -------
    offspring : dict
        New individual
    '''
    offspring = {}
    parent1 = encode_parent(parents[0], parameters)
    parent2 = encode_parent(parents[1], parameters)
    for key in parent1:
        parent1_chromosome = parent1[key]
        parent2_chromosome = parent2[key]
        offspring_chromosome = ''
        for i, chromosome in enumerate(parent1_chromosome):
            cointoss = random.random()
            if cointoss < 0.5:
                offspring_chromosome += chromosome
            else:
                offspring_chromosome += parent2_chromosome[i]
        if mutation_chance != 0:
            offspring_chromosome = chromosome_mutate(
                offspring_chromosome, mutation_chance)
        offspring[key] = offspring_chromosome
    offspring = decode_offspring(offspring, parameters)
    offspring = mutation_fix(offspring, parameters)
    return offspring


def group_crossover(parents, parameters, mutation_chance=0):
    '''Crossover of parents based on grouping.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit

    Returns
    -------
    offspring : dict
        New individual
    '''

    parent1, true_corr = grouping(parents[0], parameters)
    parent2, true_corr = grouping(parents[1], parameters)
    offspring = []
    for i, group in enumerate(parent1):
        cointoss = random.random()
        if cointoss < 0.5:
            if cointoss < (mutation_chance/2):
                mutated = group_mutate(
                    group,
                    mutation_chance,
                    cointoss,
                    true_corr[i]
                )
                offspring.append(mutated)
            else:
                offspring.append(group)
        else:
            if cointoss > (1 - float(mutation_chance) / 2):
                mutated = group_mutate(
                    parent2[i],
                    mutation_chance,
                    (1 - cointoss),
                    true_corr[i]
                )
                offspring.append(mutated)
            else:
                offspring.append(parent2[i])
    offspring = degroup(offspring)
    offspring = mutation_fix(offspring, parameters)
    return offspring


# MUTATION FUNCTIONS
def chromosome_mutate(chromosome, mutation_chance):
    '''Mutation of a single chromosome.

    Parameters
    ----------
    chromosome : string
        A single chromosome of an individual to be mutated
    mutation_chance : float
        Probability of mutation for a single bit in the chromosome

    Returns
    -------
    mutated_chromosome : string
        A mutated chromosome
    '''
    mutated_chromosome = ''
    for gene in chromosome:
        if random.random() < mutation_chance:
            try:
                mutated_chromosome += str(abs(int(gene) - 1))
            except ValueError:
                pass
        else:
            mutated_chromosome += gene
    return mutated_chromosome


def group_mutate(group, mutation_chance, cointoss, pos_corr):
    '''Mutation of a single group.

    Parameters
    ----------
    group : dict
        Group of parameter values to be mutated together
    mutation_chance : float
        Probability of mutation occuring
    cointoss : float
        Random number indicating current probability value
    pos_corr : bool
        Group is correlated either positively (True)
        or negatively (False)

    Returns
    -------
    mutation : dict
        Group of mutated parameter values
    '''
    mutation = {}
    if pos_corr == 1:
        for i, key in enumerate(group):
            if random.random() < 0.5:
                mutation[key] = (group[key]
                                 + (mutation_chance - cointoss)*group[key])
            else:
                mutation[key] = (group[key]
                                 - (mutation_chance - cointoss)*group[key])
    else:
        for i, key in enumerate(group):
            if random.random() < 0.5:
                mutation[key] = (
                    group[key]
                    + ((-1)**i)*(mutation_chance - cointoss)*group[key]
                )
            else:
                mutation[key] = (
                    group[key]
                    - ((-1)**i)*(mutation_chance - cointoss)*group[key]
                )
    return mutation


def mutation_fix(offspring, parameters):
    '''
    Fixes mutated offspring in case of parameter violation.

    Parameters
    ---------
    offspring : dict
        Mutated offspring
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    offspring : dict
        Offspring with fixed parameter values
    '''
    for parameter in parameters:
        key = parameter['p_name']
        if 'exp' not in parameter:
            parameter['exp'] = 0
        if 'true_int' not in parameter:
            parameter['true_int'] = 0
        if parameter['exp'] == 1:
            if offspring[key] > 1:
                offspring[key] = 1
            elif offspring[key] < 0:
                offspring[key] = 0
        else:
            if parameter['true_int'] == 1:
                offspring[key] = int(round(offspring[key]))
            if offspring[key] < parameter['range_start']:
                offspring[key] = parameter['range_start']
            elif offspring[key] > parameter['range_end']:
                offspring[key] = parameter['range_end']
    return offspring


# ENCODING FUNCTIONS
def int_encoding(value):
    '''Encode an int into a bit string.

    Parameters
    ----------
    value : int
        Integer to be encoded

    Returns
    -------
    encoded : string
        Encoded value of the integer
    '''
    encoded = '{:032b}'.format(value)
    return encoded


def float_encoding(value, accuracy=5):
    '''Encode a float into a bit string.

    Parameters
    ----------
    value : float
        Float to be encoded
    accuracy : int
        Number of decimals to be preserved; default is 5

    Returns
    -------
    encoded : string
        Encoded value of the float with a given accuracy
    '''
    integer = int(round(value * (10 ** accuracy)))
    encoded = '{:032b}'.format(integer)
    return encoded


def int_decoding(encoded):
    '''Decode an encoded int back into a n int.

    Parameters
    ----------
    encoded : string
        Encoded value

    Returns
    -------
    decoded : int
        Decoded value of the int
    '''
    decoded = int(encoded, 2)
    return decoded


def float_decoding(encoded, accuracy=5):
    '''Decode an encoded float back into a float.

    Parameters
    ----------
    encoded : string
        Encoded value
    accuracy : int
        Number of decimals preserved in the float; default is 5

    Returns
    -------
    value : float
        Decoded value of the float
    '''
    decoded = int(encoded, 2)
    value = float(decoded) / (10 ** accuracy)
    return value


def encode_parent(parent, parameters):
    '''Encode an individual for crossover and mutation.

    Parameters
    ----------
    parent : dict
        Individual to be encoded
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    encoded_parent : dict
        Individual with encoded values
    '''
    encoded_parent = {}
    for parameter in parameters:
        key = parameter['p_name']
        try:
            true_int = parameter['true_int']
        except KeyError:
            true_int = 0
        if true_int == 1:
            encoded_parent[key] = int_encoding(parent[key])
        else:
            encoded_parent[key] = float_encoding(parent[key])
    return encoded_parent


def decode_offspring(offspring, parameters):
    '''Decode an individual.

    Parameters
    ----------
    offspring : dict
        Individual to be decoded
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    decoded_offspring : dict
        Individual with decoded values
    '''

    decoded_offspring = {}
    for parameter in parameters:
        key = parameter['p_name']
        try:
            true_int = parameter['true_int']
        except KeyError:
            true_int = 0
        if true_int == 1:
            decoded_offspring[key] = int_decoding(offspring[key])
        else:
            decoded_offspring[key] = float_decoding(offspring[key])
    return decoded_offspring


# GROUPING FUNCTIONS
def grouping(parent, parameters):
    '''Group elements in a parent for crossover

    Parameters
    ----------
    parent : dict
        Parent to be grouped
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    grouped_parent : list
        Parent with grouped parameter values
    true_corr : list
        Correlation for each group
    '''
    grouped_parent = []
    group = {}
    true_corr = []
    for i, param in enumerate(parameters):
        key = param['p_name']
        group[key] = parent[key]
        if i + 1 > (len(parameters) - 1):
            grouped_parent.append(group)
            true_corr.append(param['true_corr'])
        elif parameters[i]['group_nr'] != parameters[i + 1]['group_nr']:
            grouped_parent.append(group)
            true_corr.append(param['true_corr'])
            group = {}
    return grouped_parent, true_corr


def degroup(offspring):
    '''Degroup elements in offspring after crossover

    Parameters
    ----------
    offspring : list
        Grouped individual to be degrouped

    Returns
    degrouped_offspring : dict
        Degrouped individual
    '''
    degrouped_offspring = {}
    for group in offspring:
        degrouped_offspring.update(group)
    return degrouped_offspring
