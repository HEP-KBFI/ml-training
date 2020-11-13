import numpy as np
import matplotlib
matplotlib.use('agg')
from machineLearning.machineLearning import evaluation_tools as et
import glob
import os
import json
import matplotlib.pyplot as plt


class Particle():

    def __init__(self, hyperparameter_info, iterations):
        self.confidence_coefficients = {'c_max': 1.62, 'w': 0.8, 'w2': 0.4}
        self.set_inertial_weight_step(iterations)
        self.hyperparameter_info = hyperparameter_info
        self.initialize_hyperparameters()
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

    def initialize_hyperparameters(self):
        self.hyperparameters = {}
        for key in self.hyperparameter_info.keys():
            if bool(self.hyperparameter_info[key]['int']):
                value = np.random.randint(
                    low=self.hyperparameter_info[key]['min'],
                    high=self.hyperparameter_info[key]['max']
                )
            else:
                value = np.random.uniform(
                    low=self.hyperparameter_info[key]['min'],
                    high=self.hyperparameter_info[key]['max']
                )
            if bool(self.hyperparameter_info[key]['exp']):
                value = np.exp(value)
            self.hyperparameters[key] = value

    def next_iteration(self):
        self.update_location()
        self.update_speeds()
        self.track_history()
        self.confidence_coefficients['w'] -= self.weight_step

    def set_hyperparameter_values(self, hyperparameters):
        for key in self.keys:
            self.hyperparameters[key] = hyperparameters[key]

class ParticleSwarm:
    def __init__(
            self, settings, fitness_function,
            hyperparameter_info, continuation, output_dir
    ):
        self.continuation = continuation
        self.output_dir = output_dir
        self.settings = settings
        self.fitness_function = fitness_function
        self.hyperparameter_info = hyperparameter_info
        self.compactnesses = []
        self.global_bests = []
        self.global_best = 0
        self.swarm = self.createSwarm()
        if self.continuation:
            hyperparameter_sets = get_iteration_info(
                self.output_dir, 0, self.settings)[1]
            self.createSetSwarm(hyperparameter_sets)

    def createSwarm(self):
        particle_swarm = []
        for i in range(self.settings['sample_size']):
            single_particle = Particle(
                self.hyperparameter_info, self.settings['iterations'])
            particle_swarm.append(single_particle)
        return particle_swarm

    def createSetSwarm(self, hyperparameter_sets):
        for particle, hyperparameters in zip(self.swarm, hyperparameter_sets):
            particle.set_hyperparameter_values(hyperparameters)

    def espionage(self):
        for particle in self.swarm:
            informants = np.random.choice(
                self.swarm, self.settings['nr_informants']
            )
            best_fitnesses, best_locations = self.get_fitnesses_and_location(
                informants)
            particle.gather_intelligence(best_locations, best_fitnesses)

    def get_fitnesses_and_location(self, group):
        best_locations = []
        best_fitnesses = []
        for particle in group:
            best_fitnesses.append(particle.personal_best_fitness)
            best_locations.append(particle.personal_best)
        return best_fitnesses, best_locations

    def set_particle_fitnesses(self, fitnesses, initial=False):
        for particle, fitness in zip(self.swarm, fitnesses):
            if initial:
                particle.set_initial_bests(fitness)
            else:
                particle.set_fitness(fitness)

    def find_best_hyperparameters(self):
        best_fitnesses, best_locations = self.get_fitnesses_and_location(
            self.swarm)
        index = np.argmax(best_fitnesses)
        best_fitness = best_fitnesses[index]
        best_location = best_locations[index]
        return best_fitness, best_location

    def check_global_best(self):
        for particle in self.swarm:
            if particle.fitness > self.global_best:
                self.global_best = particle.fitness
        self.global_bests.append(self.global_best)

    def particleSwarmOptimization(self):
        iteration = 0
        if self.continuation:
            last_complete_iteration = collect_iteration_particles(self.output_dir)
            fitnesses, all_locations = get_iteration_info(
                self.output_dir, iteration, self.settings)
        else:
            all_locations = [particle.hyperparameters for particle in self.swarm]
            fitnesses = self.fitness_function(all_locations, self.settings)
        self.set_particle_fitnesses(fitnesses, initial=True)
        self.check_global_best()
        for particle in self.swarm:
            particle.next_iteration()
        compactness = et.calculate_compactness(all_locations)
        self.compactnesses.append(compactness)
        not_clustered = True
        plot_progress(iteration, self.compactnesses, 'compactness', self.output_dir)
        plot_progress(iteration, self.global_bests, 'global_best', self.output_dir)
        iteration = 1
        while iteration <= self.settings['iterations'] and not_clustered:
            print('::::::: Iteration: ' + str(iteration) + ' ::::::::')
            self.espionage()
            all_locations = [particle.hyperparameters for particle in self.swarm]
            if self.continuation and iteration <= last_complete_iteration:
                fitnesses, all_locations = get_iteration_info(
                    self.output_dir, iteration, self.settings)
            else:
                fitnesses = self.fitness_function(all_locations, self.settings)
            self.set_particle_fitnesses(fitnesses)
            self.check_global_best()
            plot_progress(iteration, self.compactnesses, 'compactness', self.output_dir)
            plot_progress(iteration, self.global_bests, 'global_best', self.output_dir)
            for particle in self.swarm:
                particle.next_iteration()
            compactness = et.calculate_compactness(all_locations)
            self.compactnesses.append(compactness)
            print(' --- Compactness: ' + str(compactness) + '---')
            not_clustered = compactness > self.settings['compactness_threshold']
            iteration += 1
        best_fitness, best_location = self.find_best_hyperparameters()
        print('Best location is: ' + str(best_location))
        print('Best_fitness is: ' + str(best_fitness))
        return best_location, best_fitness


def collect_iteration_particles(iteration_dir):
    iteration_paths = os.path.join(iteration_dir, 'previous_files', 'iteration_*')
    all_iterations = glob.glob(iteration_paths)
    return check_last_iteration_completeness(all_iterations, iteration_dir)


def check_last_iteration_completeness(all_iterations, iteration_dir):
    iteration_nrs = [int(iteration.split('_')[-1]) for iteration in all_iterations]
    iteration_nrs.sort()
    last_iteration = os.path.join(iteration_dir, 'iteration_' + str(iteration_nrs[-1]))
    all_particles_wildcard = os.path.join(last_iteration, '*')
    for path in glob.glob(all_particles_wildcard):
        parameter_file = os.path.join(path, 'parameters.json')
        score_file = os.path.join(path, 'score.json')
        if not os.path.exists(parameter_file):
            return iteration_nrs[-2]
        if not os.path.exists(score_file):
            return iteration_nrs[-2]
    return iteration_nrs[-1]


def get_iteration_info(output_dir, iteration, settings):
    number_particles = settings['sample_size']
    iteration_dir = os.path.join(
        output_dir, 'previous_files', 'iteration_' + str(iteration))
    fitnesses = []
    parameters_list = []
    for particle in range(number_particles):
        particle_dir = os.path.join(iteration_dir, str(particle))
        score_file = os.path.join(particle_dir, 'score.json')
        parameter_file = os.path.join(particle_dir, 'parameters.json')
        with open(score_file, 'rt') as inFile:
            fitness = json.load(inFile)['d_roc']
        with open(parameter_file, 'rt') as inFile:
            parameters = json.load(inFile)
        fitnesses.append(fitness)
        parameters_list.append(parameters)
    return fitnesses, parameters_list


def plot_progress(iteration, y_values, variable_name, output_dir):
    iterations = np.arange(iteration + 1)
    plt.plot(iterations, y_values)
    plt.xlabel(iterations)
    plt.ylabel(variable_name)
    plt.grid()
    output_path = os.path.join(output_dir, variable_name + '_progress.png')
    plt.savefig(output_path, bbox_inches='tight')
