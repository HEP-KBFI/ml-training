'''Testing the crossover functions for the genetic algorithm
'''
from machineLearning.machineLearning import ga_crossover as gc


PARAMETERS = [
    {
        'p_name': 'a',
        'range_start': 0,
        'range_end': 10,
        'true_int': 1,
        'group_nr': 1,
        'true_corr': 1
    },
    {
        'p_name': 'b',
        'range_start': 1,
        'range_end': 10,
        'true_int': 0,
        'group_nr': 1,
        'true_corr': 1
    },
    {
        'p_name': 'c',
        'range_start': 0,
        'range_end': 1,
        'true_int': 0,
        'group_nr': 2,
        'true_corr': 0
    },
    {
        'p_name': 'd',
        'range_start': 5,
        'range_end': 25,
        'true_int': 1,
        'group_nr': 2,
        'true_corr': 0
    }
]


PARENTS = [
    {
        'a': 8,
        'b': 1.62284,
        'c': 0.31443,
        'd': 15
    },
    {
        'a': 4,
        'b': 9.43930,
        'c': 0.92191,
        'd': 24
    }
]


def test_kpoint_crossover():
    '''Testing k-point crossover function'''
    result = gc.kpoint_crossover(PARENTS, PARAMETERS)
    for parent in PARENTS:
        for key in parent:
            assert key in result, 'test_kpoint_crossover failed'
        assert len(result) == len(parent), 'test_kpoint_crossover failed'


def test_uniform_crossover():
    '''Testing uniform crossover function'''
    result = gc.uniform_crossover(PARENTS, PARAMETERS)
    for parent in PARENTS:
        for key in parent:
            assert key in result, 'test_uniform_crossover failed'
        assert len(result) == len(parent), 'test_uniform_crossover failed'


def test_group_crossover():
    '''Testing group crossover function'''
    result = gc.group_crossover(PARENTS, PARAMETERS)
    for parent in PARENTS:
        for key in parent:
            assert key in result, 'test_group_crossover failed'
        assert len(result) == len(parent), 'test_uniform_crossover failed'


def test_chromosome_mutate():
    '''Testing chromosome mutate function'''
    initial = '0101010101'

    # Mutation chance = 0
    result1 = gc.chromosome_mutate(initial, 0)
    assert initial == result1, 'test_chromosome_mutate failed'
    for key in initial:
        assert key in result1, 'test_chromosome_mutate failed'
    assert len(initial) == len(result1), 'test_chromosome_mutate failed'

    # Mutation chance = 0.5
    result2 = gc.chromosome_mutate(initial, 0.5)
    for key in initial:
        assert key in result2, 'test_chromosome_mutate failed'
    assert len(initial) == len(result2), 'test_chromosome_mutate failed'


def test_group_mutate():
    '''Testing group mutate function'''
    initial = {'a': 8, 'b': 1.62284}

    # Mutation chance = 0
    result1 = gc.group_mutate(initial, 0, 0, 1)
    assert initial == result1, 'test_group_mutate failed'
    for key in initial:
        assert key in result1, 'test_group_mutate failed'
    assert len(result1) == len(initial), 'test_group_mutate failed'

    # Mutation chance = 0.5
    result2 = gc.group_mutate(initial, 0.5, 0.25, 0)
    for key in initial:
        assert key in result2, 'test_group_mutate failed'
    assert len(result2) == len(initial), 'test_group_mutate failed'


def test_mutation_fix():
    '''Testing mutation fix function'''
    initial = {'a': 10.01, 'b': 0.99999, 'c': 1.00001, 'd': 9.43930}
    expected = {'a': 10, 'b': 1, 'c': 1, 'd': 9}
    result = gc.mutation_fix(initial, PARAMETERS)
    assert result == expected, 'test_mutation_fix failed'


def test_int_encoding():
    '''Testing integer encoding function'''
    initial = [2, 10, 500, 3000, 54000]
    expected = [
        '00000000000000000000000000000010',
        '00000000000000000000000000001010',
        '00000000000000000000000111110100',
        '00000000000000000000101110111000',
        '00000000000000001101001011110000'
    ]
    results = []
    for number in initial:
        results.append(gc.int_encoding(number))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_int_encoding failed'


def test_float_encoding():
    '''Testing float encoding function'''
    initial = [0.1, 0.002, 0.00003, 0.0000004, 55.55]
    expected = [
        '00000000000000000010011100010000',
        '00000000000000000000000011001000',
        '00000000000000000000000000000011',
        '00000000000000000000000000000000',
        '00000000010101001100001100111000',
    ]
    results = []
    for number in initial:
        results.append(gc.float_encoding(number))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_float_encoding failed'


def test_int_decoding():
    '''Testing integer decoding function'''
    initial = [
        '00000000000000000000000000000010',
        '00000000000000000000000000001010',
        '00000000000000000000000111110100',
        '00000000000000000000101110111000',
        '00000000000000001101001011110000'
    ]
    expected = [2, 10, 500, 3000, 54000]
    results = []
    for code in initial:
        results.append(gc.int_decoding(code))
    for i, result in enumerate(results):
        assert result == expected[i], 'test_int_decoding failed'


def test_float_decoding():
    '''Testing float decoding function'''
    initial = [
        '00000000000000000010011100010000',
        '00000000000000000000000011001000',
        '00000000000000000000000000000011',
        '00000000000000000000000000000000',
        '00000000010101001100001100111000'
    ]
    expected = [0.1, 0.002, 0.00003, 0.0, 55.55]
    results = []
    for code in initial:
        results.append(gc.float_decoding(code))
    for i, result in enumerate(results):
        assert result == expected[i], 'float_int_decoding failed'


def test_encode_parent():
    '''Testing encoding parent function'''
    initial = PARENTS[0]
    expected = {
        'a': '00000000000000000000000000001000',
        'b': '00000000000000100111100111101100',
        'c': '00000000000000000111101011010011',
        'd': '00000000000000000000000000001111'
    }
    result = gc.encode_parent(initial, PARAMETERS)
    assert result == expected, 'test_encode_parent failed'


def test_decode_offspring():
    '''Testing decoding offspring function'''
    initial = {
        'a': '00000000000000000000000000001000',
        'b': '00000000000000100111100111101100',
        'c': '00000000000000000111101011010011',
        'd': '00000000000000000000000000001111'
    }
    expected = PARENTS[0]
    result = gc.decode_offspring(initial, PARAMETERS)
    assert result == expected, 'test_decode_offspring failed'


def test_grouping():
    '''Testing grouping function'''
    initial = PARENTS[0]
    expected1 = [{'a': 8, 'b': 1.62284}, {'c': 0.31443, 'd': 15}]
    expected2 = [1, 0]
    result1, result2 = gc.grouping(initial, PARAMETERS)
    assert result1 == expected1, 'test_grouping failed'
    assert result2 == expected2, 'test_grouping failed'


def test_degroup():
    '''Testing degrouping function'''
    initial = [{'a': 8, 'b': 1.62284}, {'c': 0.31443, 'd': 15}]
    expected = PARENTS[0]
    result = gc.degroup(initial)
    assert result == expected, 'test_degroup failed'
