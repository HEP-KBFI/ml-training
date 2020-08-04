import numpy as np
from machineLearning.machineLearning import evaluation_tools as et


class Particle():

    def __init__(self, value_dicts, iterations):
        self.confidence_coefficients = {'c_max': 1.62, 'w': 0.8, 'w2': 0.4}
        self.set_inertial_weight_step(iterations)
        self.set_parameter_info(value_dicts)
        self.initialize_hyperparameters(value_dicts)
        self.keys = self.hyperparameters.keys()
        self.initialize_speeds()
        self.personal_best_history = []
        self.personal_best_fitness_history = []
        self.fitness_history = []
        self.location_history = []
        self.total_iterations = iterations
        self.iteration = 0

    def set_inertial_weight_step(self, iterations):
        range_size = (
            self.confidence_coefficients['w'] - \
            self.confidence_coefficients['w2']
        )
        self.weight_step = range_size / iterations

    def initialize_speeds(self):
        self.speed = {}
        for key in self.keys:
            v_max = (
                self.hyperparameter_info[key]['max'] - \
                self.hyperparameter_info[key]['min'] / 4
            )
            self.speed[key] = np.random.uniform() * v_max

    def set_parameter_info(self, value_dicts):
        self.hyperparameter_info = {}
        for value_info in value_dicts:
            value_copy = value_info.copy()
            key = value_copy.pop('parameter')
            self.hyperparameter_info[key] = value_copy

    def set_fitness(self, fitness):
        self.fitness = fitness
        if self.fitness > self.personal_best_fitness:
            self.set_personal_best()
        if self.fitness > self.global_best_fitness:
            self.set_global_best(self.hyperparameters, self.fitness)

    def set_personal_best(self):
        self.personal_best = self.hyperparameters.copy()
        self.personal_best_fitness = float(self.fitness)

    def set_global_best(self, hyperparameters, fitness):
        self.global_best = hyperparameters.copy()
        self.global_best_fitness = float(fitness)


    def set_initial_bests(self, fitness):
        self.fitness = fitness
        self.set_personal_best()
        self.set_global_best(self.hyperparameters, self.fitness)

    def update_speeds(self):
        for key in self.keys:
            rand1 = np.random.uniform()
            rand2 = np.random.uniform()
            cognitive_component = self.confidence_coefficients['c_max'] * rand1 * (
                self.personal_best[key] - self.hyperparameters[key])
            social_component = self.confidence_coefficients['c_max'] * rand2 * (
                self.global_best[key] - self.hyperparameters[key])
            inertial_component = (
                self.confidence_coefficients['w'] * self.speed[key]
            )
            self.speed[key] = (
                cognitive_component
                + social_component
                + inertial_component
            )

    def update_location(self):
        for key in self.keys:
            self.hyperparameters[key] += self.speed[key]
            if self.hyperparameter_info[key]['exp'] == 1:
                max_value = np.exp(self.hyperparameter_info[key]['max'])
                min_value = np.exp(self.hyperparameter_info[key]['min'])
            else:
                max_value = self.hyperparameter_info[key]['max']
                min_value = self.hyperparameter_info[key]['min']
            if self.hyperparameters[key] > max_value:
                self.hyperparameters[key] = max_value
                self.speed[key] = 0
            if self.hyperparameters[key] < min_value:
                self.hyperparameters[key] = min_value
                self.speed[key] = 0
            if self.hyperparameter_info[key]['int'] == 1:
                self.hyperparameters[key] = int(np.ceil(self.hyperparameters[key]))

    def gather_intelligence(self, locations, fitnesses):
        index = np.argmax(fitnesses)
        max_fitness = max(fitnesses)
        if max_fitness > self.global_best_fitness:
            self.set_global_best(locations[index], fitnesses[index])

    def track_history(self):
        self.personal_best_history.append(self.personal_best)
        self.personal_best_fitness_history.append(self.personal_best_fitness)
        self.fitness_history.append(self.fitness)
        self.location_history.append(self.hyperparameters)

    def initialize_hyperparameters(self, value_dicts):
        self.hyperparameters = {}
        for parameter_info in value_dicts:
            if bool(parameter_info['int']):
                value = np.random.randint(
                    low=parameter_info['min'],
                    high=parameter_info['max']
                )
            else:
                value = np.random.uniform(
                    low=parameter_info['min'],
                    high=parameter_info['max']
                )
            if bool(parameter_info['exp']):
                value = np.exp(value)
            self.hyperparameters[str(parameter_info['parameter'])] = value

    def next_iteration(self):
        self.update_location()
        self.update_speeds()
        self.track_history()
        self.confidence_coefficients['w'] -= self.weight_step



############################################################################


def particleSwarmOptimization(settings, fitness_function, value_dicts):
    number_particles = settings['sample_size']
    number_iterations = settings['iterations']
    number_informants = settings['nr_informants']
    iteration = 0
    particle_swarm = createSwarm(
        number_particles, value_dicts, fitness_function, number_iterations)
    all_locations = [particle.hyperparameters for particle in particle_swarm]
    fitnesses = fitness_function(all_locations, settings)
    set_particle_fitnesses(particle_swarm, fitnesses, initial=True)
    for particle in particle_swarm:
        particle.next_iteration()
    not_clustered = True
    while iteration <= number_iterations and not_clustered:
        print('::::::: Iteration: ' + str(iteration) + ' ::::::::')
        espionage(number_informants, particle_swarm)
        all_locations = [particle.hyperparameters for particle in particle_swarm]
        fitnesses = fitness_function(all_locations, settings)
        set_particle_fitnesses(particle_swarm, fitnesses)
        for particle in particle_swarm:
            particle.next_iteration()
        compactness = et.calculate_compactness(all_locations)
        print(' --- Compactness: ' + str(compactness) + '---')
        not_clustered = compactness < settings['compactness_threshold']
        iteration += 1
    best_fitness, best_location = find_best_hyperparameters(particle_swarm)
    print('Best location is: ' + str(best_location))
    print('Best_fitness is: ' + str(best_fitness))
    return best_fitness


##############################################################################


def createSwarm(
        number_particles, value_dicts,
        fitness_function, iterations
):
    particle_swarm = []
    for i in range(number_particles):
        single_particle = Particle(value_dicts, iterations)
        particle_swarm.append(single_particle)
    return particle_swarm


def espionage(number_informants, particle_swarm):
    for particle in particle_swarm:
        informants = np.random.choice(particle_swarm, number_informants)
        best_fitnesses, best_locations  = get_fitnesses_and_location(informants)
        particle.gather_intelligence(best_locations, best_fitnesses)


def get_fitnesses_and_location(particles):
    best_locations = []
    best_fitnesses = []
    for particle in particles:
        best_fitnesses.append(particle.personal_best_fitness)
        best_locations.append(particle.personal_best)
    return best_fitnesses, best_locations


def set_particle_fitnesses(particle_swarm, fitnesses, initial=False):
    for particle, fitness in zip(particle_swarm, fitnesses):
        if initial:
            particle.set_initial_bests(fitness)
        else:
            particle.set_fitness(fitness)


def find_best_hyperparameters(particle_swarm):
    best_fitnesses, best_locations = get_fitnesses_and_location(particle_swarm)
    index = np.argmax(best_fitnesses)
    best_fitness = best_fitnesses[index]
    best_location = best_locations[index]
    return best_fitness, best_location