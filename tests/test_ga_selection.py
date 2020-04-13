'''Testing the selection functions for the genetic algorithm'''
from machineLearning.machineLearning import ga_selection as gs

POPULATION = [1, 2, 3]
FITNESSES = [0.4, 0.6, 0.8]
PROBABILITIES = [0.2, 0.3, 0.5]


def test_tournament():
    '''Testing tournament selection'''
    result = gs.tournament(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_tournament failed'
    for member in result:
        assert member in POPULATION, 'test_tournament failed'


def test_roulette():
    '''Testing roulette wheel selection'''
    result = gs.roulette(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_roulette failed'
    for member in result:
        assert member in POPULATION, 'test_roulette failed'


def test_rank():
    '''Testing rank selection'''
    result = gs.rank(POPULATION, FITNESSES)
    assert len(result) == 2, 'test_rank failed'
    for member in result:
        assert member in POPULATION, 'test_rank failed'


def test_normalize():
    '''Testing the function normalizing the fitness scores'''
    result = gs.normalize(FITNESSES)
    assert sum(result) == 1, 'test_normalize failed'


def test_wheel_parents():
    '''Testing selection of parents via a generated wheel'''
    result = gs.wheel_parents(POPULATION, PROBABILITIES)
    assert len(result) == 2, 'test_wheel failed'
    for member in result:
        assert member in POPULATION, 'test_wheel failed'