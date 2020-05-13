''' Tools necessary for running the Particle Swarm Optimization algorithm.
'''
import numbers
import numpy as np
import os
from machineLearning.machineLearning import universal_tools as ut
from machineLearning.machineLearning import evaluation_tools as et


def run_pso(
        value_dicts,
        calculate_fitnesses,
        hyperparameter_sets,
        output_dir
):
    '''Performs the whole particle swarm optimization. Pay attention that the
    best fitness has the maximum value, not the minimum. (multiply by -1 if
    needed)

    Parameters:
    ----------
    value_dicts : list of dicts
        Info about every variable that is to be optimized
    calculate_fitness : method
        Function that calculates the fitness and returns the  score
    hyperparameter_sets : list of dicts
        The parameter-sets of all particles.
    output_dir : str
        Path to the directory of the output

    Returns:
    -------
    best_hyperparameters : dict
        Best hyperparameters found.
    '''
    print(':::::::: Initializing :::::::::')
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = ut.read_settings(settings_dir, 'global')
    pso_settings = ut.read_settings(settings_dir, 'pso')
    inertial_weight, inertial_weight_step = get_weight_step(pso_settings)
    iteration = 1
    new_hyperparameter_sets = hyperparameter_sets
    personal_bests = {}
    compactness = et.calculate_compactness(hyperparameter_sets)
    fitnesses = calculate_fitnesses(hyperparameter_sets, global_settings)
    personal_bests = hyperparameter_sets
    best_fitnesses = fitnesses
    index = np.argmax(fitnesses)
    best_hyperparameters = hyperparameter_sets[index]
    best_fitness = fitnesses[index]
    current_speeds = initialize_speeds(hyperparameter_sets)
    max_iterations_not_reached = True
    not_clustered = True
    while max_iterations_not_reached and not_clustered:
        print('::::::: Iteration: ' + str(iteration) + ' ::::::::')
        hyperparameter_sets = new_hyperparameter_sets
        compactness = et.calculate_compactness(hyperparameter_sets)
        print(' --- Compactness: ' + str(compactness) + ' ---')
        fitnesses = calculate_fitnesses(hyperparameter_sets, global_settings)
        best_fitnesses = find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = calculate_personal_bests(
            fitnesses, best_fitnesses, hyperparameter_sets, personal_bests)
        weight_dict = {
            'c1': pso_settings['c1'],
            'c2': pso_settings['c2'],
            'w': inertial_weight}
        new_hyperparameter_sets, current_speeds = prepare_new_day(
            personal_bests, hyperparameter_sets,
            best_hyperparameters,
            current_speeds, value_dicts,
            weight_dict
        )
        index = np.argmax(fitnesses)
        if best_fitness < max(fitnesses):
            best_hyperparameters = hyperparameter_sets[index]
            best_fitness = fitnesses[index]
        inertial_weight += inertial_weight_step
        iteration += 1
        max_iterations_not_reached = iteration <= pso_settings['iterations']
        not_clustered = pso_settings['compactness_threshold'] < compactness
    return best_hyperparameters


def get_weight_step(pso_settings):
    '''Calculates the step size of the inertial weight

    Parameters:
    ----------
    pso_settings : dict
        PSO settings

    Returns:
    -------
    inertial_weight : float
        inertial weight
    inertial_weight_step : float
        Step size of the inertial weight
    '''
    inertial_weight = np.array(pso_settings['w_init'])
    inertial_weight_fin = np.array(pso_settings['w_fin'])
    inertial_weight_init = np.array(pso_settings['w_init'])
    weight_difference = float(inertial_weight_fin - inertial_weight_init)
    inertial_weight_step = weight_difference / pso_settings['iterations']
    return inertial_weight, inertial_weight_step


def check_numeric(variables):
    '''Checks whether the variable is numeric

    Parameters:
    ----------
    variables : list

    Returns:
    -------
    decision : bool
        Decision whether the list of variables contains non-numeric values
    '''
    nr_nonnumeric = 0
    decision = False
    for variable in variables:
        if not isinstance(variable, numbers.Number):
            nr_nonnumeric += 1
    if nr_nonnumeric > 0:
        decision = True
    return decision


def calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        hyperparameter_sets,
        personal_bests
):
    '''Find best parameter-set for each particle

    Parameters:
    ----------
    fitnesses : list
        List of current iteration fitnesses for each particle
    best_fitnesses : list
        List of best fitnesses for each particle
    hyperparameter_sets : list of dicts
        Current parameters of the last iteration for each particle
    personal_bests : list of dicts
        Best parameters (with highest fitness) for each particle so far

    Returns:
    -------
    new_dicts : list of dicts
        Personal best parameter-sets for each particle
    '''
    new_dicts = []
    for fitness, best_fitness, hyperparameters, personal_best in zip(
            fitnesses, best_fitnesses, hyperparameter_sets, personal_bests):
        non_numeric = check_numeric(
            [fitness, best_fitness])
        if non_numeric:
            raise TypeError
        if fitness > best_fitness:
            new_dicts.append(hyperparameters)
        else:
            new_dicts.append(personal_best)
    return new_dicts


def calculate_new_position(
        speeds,
        hyperparameter_sets,
        value_dicts
):
    '''Calculates the new parameters for the next iteration

    Parameters:
    ----------
    speeds : list of dicts
        Current speed in each parameter direction for each particle
    hyperparameter_sets : list of dicts
        Current parameter-sets of all particles
    value_dicts : list of dicts
        Info about every variable that is to be optimized

    Returns:
    -------
    new_values : list of dicts
        New parameters to be used in the next iteration
    '''
    new_values = []
    for speed, hyperparameters in zip(speeds, hyperparameter_sets):
        new_value = {}
        for parameter in value_dicts:
            key = parameter['p_name']
            if bool(parameter['true_int']):
                new_value[key] = int(np.ceil(
                    hyperparameters[key] + speed[key]))
            else:
                new_value[key] = hyperparameters[key] + speed[key]
            if parameter['exp'] == 1:
                parameter_start = np.exp(parameter['range_start'])
                parameter_end = np.exp(parameter['range_end'])
            elif parameter['exp'] == 0:
                parameter_start = parameter['range_start']
                parameter_end = parameter['range_end']
            else:
                print('Check the "exp" parameter in "xgb or nn parameter file"')
            if parameter_start > new_value[key]:
                new_value[key] = parameter_start
            elif parameter_end < new_value[key]:
                new_value[key] = parameter_end
        new_values.append(new_value)
    return new_values


def calculate_new_speed(
        personal_bests,
        hyperparameter_sets,
        best_parameters,
        current_speeds,
        weight_dict
):
    '''Calculates the new speed in each parameter direction for all particles

    Parameters:
    ----------
    personal_bests : list of dicts
        Best parameters for each individual particle
    hyperparameter_sets : list of dicts
        Current iteration parameters for each particle
    current_speeds : list of dicts
        Speed in every parameter direction for each particle
    weight_dict : dict
        dictionary containing the normalized weights [w: inertial weight,
        c1: cognitive weight, c2: social weight]

    Returns:
    -------
    new_speeds : list of dicts
        The new speed of the particle in each parameter direction
    '''
    new_speeds = []
    for personal, current, inertia in zip(
            personal_bests, hyperparameter_sets, current_speeds
    ):
        new_speed = {}
        for key in current:
            rand1 = np.random.uniform()
            rand2 = np.random.uniform()
            cognitive_component = weight_dict['c1'] * rand1 * (
                personal[key] - current[key])
            social_component = weight_dict['c2'] * rand2 * (
                best_parameters[key] - current[key])
            inertial_component = weight_dict['w'] * inertia[key]
            new_speed[key] = (
                cognitive_component
                + social_component
                + inertial_component
            )
        new_speeds.append(new_speed)
    return new_speeds


def initialize_speeds(hyperparameter_sets):
    '''Initializes the speeds in the beginning to be 0

    Parameters:
    ----------
    hyperparameter_sets : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    speeds : list of dicts
        Speeds of all particles in all parameter directions. All are 0
    '''
    speeds = []
    for hyperparameters in hyperparameter_sets:
        speed = {}
        for key in hyperparameters:
            speed[key] = 0
        speeds.append(speed)
    return speeds


def find_best_fitness(fitnesses, best_fitnesses):
    '''Compares the current best fitnesses with the current ones and
    substitutes one if it finds better

    Parameters:
    ----------
    fitnesses : list
        List of current iteration fitnesses
    best_fitnesses : list
        List of the best found fitnesses

    Returns:
    -------
    new_best_fitnesses : list
        List of best fitnesses taken into account the ones found current
        iteration
    '''
    new_best_fitnesses = []
    for fitness, best_fitness in zip(fitnesses, best_fitnesses):
        if fitness > best_fitness:
            new_best_fitnesses.append(fitness)
        else:
            new_best_fitnesses.append(best_fitness)
    return new_best_fitnesses


def prepare_new_day(
        personal_bests,
        hyperparameter_sets,
        best_parameters,
        current_speeds,
        value_dicts,
        weight_dict
):
    '''Finds the new new parameters to find the fitness of

    Parameters:
    ----------
    personal_bests : list of dicts
        Best parameters for each individual particle
    hyperparameter_sets : list of dicts
        Current iteration parameters for each particle
    current_speeds : list of dicts
        Speed in every parameter direction for each particle
    value_dicts : list of dicts
        Info about every variable that is to be optimized
    weight_dict : dict
        dictionary containing the normalized weights [w: inertial weight,
        c1: cognitive weight, c2: social weight]

    Returns:
    -------
    new_parameters : list of dicts
        Parameter-sets that are used in the next iteration
    current_speeds : list of dicts
        New speed of each particle
    '''
    current_speeds = calculate_new_speed(
        personal_bests, hyperparameter_sets, best_parameters,
        current_speeds, weight_dict
    )
    new_parameters = calculate_new_position(
        current_speeds, hyperparameter_sets, value_dicts)
    return new_parameters, current_speeds
