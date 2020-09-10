import numpy as np
from machineLearning.machineLearning import evaluation_tools as et


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
            self.hyperparameters[str(self.hyperparameter_info[key]['parameter'])] = value

    def next_iteration(self):
        self.update_location()
        self.update_speeds()
        self.track_history()
        self.confidence_coefficients['w'] -= self.weight_step


class ParticleSwarm:
    def __init__(self, settings, fitness_function, hyperparameter_info):
        self.settings = settings
        self.fitness_function = fitness_function
        self.hyperparameter_info = hyperparameter_info
        self.swarm = self.createSwarm()

    def createSwarm(self):
        particle_swarm = []
        for i in range(self.settings['sample_size']):
            single_particle = Particle(
                self.hyperparameter_info, self.settings['iterations'])
            particle_swarm.append(single_particle)
        return particle_swarm

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

    def particleSwarmOptimization(self):
        iteration = 0
        all_locations = [particle.hyperparameters for particle in self.swarm]
        fitnesses = self.fitness_function(all_locations, self.settings)
        self.set_particle_fitnesses(fitnesses, initial=True)
        for particle in self.swarm:
            particle.next_iteration()
        not_clustered = True
        while iteration <= self.settings['iterations'] and not_clustered:
            print('::::::: Iteration: ' + str(iteration) + ' ::::::::')
            self.espionage()
            all_locations = [particle.hyperparameters for particle in self.swarm]
            fitnesses = self.fitness_function(all_locations, self.settings)
            self.set_particle_fitnesses(fitnesses)
            for particle in self.swarm:
                particle.next_iteration()
            compactness = et.calculate_compactness(all_locations)
            print(' --- Compactness: ' + str(compactness) + '---')
            not_clustered = compactness > self.settings['compactness_threshold']
            iteration += 1
        best_fitness, best_location = self.find_best_hyperparameters()
        print('Best location is: ' + str(best_location))
        print('Best_fitness is: ' + str(best_fitness))
        return best_location, best_fitness