from machineLearning.machineLearning import pso_tools as pt

value_dicts = [
    {
        'p_name': 'num_boost_round',
        'range_start': 1,
        'range_end': 500,
        'true_int': 1,
        'group_nr': 1,
        'true_corr': 0,
        'exp': 0
    },
    {
        'p_name': 'learning_rate',
        'range_start': -5,
        'range_end': 0,
        'true_int': 0,
        'group_nr': 1,
        'true_corr': 0,
        'exp': 1
    },
    {
        'p_name': 'max_depth',
        'range_start': 1,
        'range_end': 10,
        'true_int': 1,
        'group_nr': 2,
        'true_corr': 0,
        'exp': 0
    },
    {
        'p_name': 'gamma',
        'range_start': 0,
        'range_end': 5,
        'true_int': 0,
        'group_nr': 2,
        'true_corr': 0,
        'exp': 0
    },
    {
        'p_name': 'min_child_weight',
        'range_start': 0,
        'range_end': 500,
        'true_int': 0,
        'group_nr': 3,
        'true_corr': 0,
        'exp': 0
    },
    {
        'p_name': 'subsample',
        'range_start': 0.8,
        'range_end': 1,
        'true_int': 0,
        'group_nr': 4,
        'true_corr': 0,
        'exp': 0
    },
    {
        'p_name': 'colsample_bytree',
        'range_start': 0.3,
        'range_end': 1,
        'true_int': 0,
        'group_nr': 5,
        'true_corr': 0,
        'exp': 0
    }
]

# def test_run_pso():

def test_get_weight_step():
    pso_settings = {'w_init': 1, 'w_fin': 0, 'iterations': 10}
    inertial_weight, inertial_weight_step = pt.get_weight_step(
        pso_settings)
    assert inertial_weight == 1
    assert inertial_weight_step == -0.1


def test_calculate_personal_bests():
    fitnesses = [0.9, 1, 0.9]
    best_fitnesses = [0.8, 0.9, 1]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    result = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    calculated_pb = pt.calculate_personal_bests(
        fitnesses,
        best_fitnesses,
        parameter_dicts,
        personal_bests
    )
    assert result == calculated_pb


def test_calculate_personal_bests2():
    fitnesses = ['a', 1, 0.9]
    best_fitnesses = [0.8, 0.9, 1]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    result = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    error = False
    try:
        calculated_pb = pt.calculate_personal_bests(
            fitnesses,
            best_fitnesses,
            parameter_dicts,
            personal_bests
        )
    except TypeError:
        error = True
    assert error == True


def test_calculate_new_position():
    parameter_dict = {
        'num_boost_round': 0,
        'learning_rate': 0,
        'max_depth': 0,
        'gamma': 0,
        'min_child_weight': 0,
        'subsample': 0,
        'colsample_bytree': 0,
    }
    parameter_dicts = [
        parameter_dict,
        parameter_dict,
        parameter_dict
    ]
    values = {
        'num_boost_round': 1,
        'learning_rate': 1,
        'max_depth': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
    }
    current_speed = {
        'num_boost_round': 1,
        'learning_rate': 1,
        'max_depth': 1,
        'gamma': 1,
        'min_child_weight': 1,
        'subsample': 1,
        'colsample_bytree': 1,
    }
    current_speeds = [
        current_speed,
        current_speed,
        current_speed
    ]
    expected = [
        values,
        values,
        values
    ]
    result = pt.calculate_new_position(
        current_speeds, parameter_dicts, value_dicts)
    assert result == expected


def test_calculate_new_speed():
    weight_dict = {
        'w': 2,
        'c1': 2,
        'c2': 2
    }
    current_speed = {'a': 1, 'b': 1, 'c': 1}
    current_speeds = [
        current_speed,
        current_speed,
        current_speed
    ]
    parameter_dicts = [
        {'a': 1, 'b': 1, 'c': 1},
        {'a': 2, 'b': 2, 'c': 2},
        {'a': 3, 'b': 3, 'c': 3}
    ]
    personal_bests = [
        {'a': 9, 'b': 9, 'c': 9},
        {'a': 8, 'b': 8, 'c': 8},
        {'a': 7, 'b': 7, 'c': 7}
    ]
    best_parameters = {'a': 2, 'b': 2, 'c': 2}
    result = pt.calculate_new_speed(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        weight_dict
    )
    assert result[0]['a'] >= 2 and result[0]['a'] <= 20
    assert result[1]['b'] >= 2 and result[1]['b'] <= 14
    assert result[2]['c'] >= 0 and result[2]['c'] <= 10


def test_calculate_new_speed2():
    weight_dict = {
        'w': 2,
        'c1': 2,
        'c2': 2
    }
    values = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    current_speeds = [1, 1, 1]
    current_values = [
        values,
        values,
        values
    ]
    pb_list = [
        values,
        values,
        values
    ]
    best_params = values
    error = False
    try:
        result = pt.calculate_new_speed(
            pb_list,
            current_values,
            best_params,
            current_speeds,
            weight_dict
        )
    except TypeError:
        error = True
    assert error == True


def test_initialize_speeds():
    parameter_dicts = [
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 3, 'b': 2, 'c': 1}
    ]
    speeds = pt.initialize_speeds(parameter_dicts)
    expected = [
        {'a': 0, 'b': 0, 'c': 0},
        {'a': 0, 'b': 0, 'c': 0}
    ]
    assert speeds == expected


def test_find_best_fitness():
    fitnesses = [0.5, 0.7, 0.1, 0.2]
    best_fitnesses = [0.4, 0.8, 0.7, 0.3]
    expected = [0.5, 0.8, 0.7, 0.3]
    result = pt.find_best_fitness(fitnesses, best_fitnesses)
    assert result == expected


def test_prepare_new_day():
    personal_bests = [
        {'foo': 1, 'bar': 2},
        {'foo': 2, 'bar': 2}
    ]
    parameter_dicts = [
        {'foo': 3, 'bar': 3},
        {'foo': 4, 'bar': 4}
    ]
    best_parameters = {'foo': 1, 'bar': 2}
    current_speeds = [
        {'foo': 0, 'bar': 0},
        {'foo': 0, 'bar': 0}
    ]
    value_dicts = [
        {
            'p_name': 'foo',
            'range_start': 1,
            'range_end': 500,
            'true_int': 0,
            'exp':0
        }
        ,
        {
            'p_name': 'bar',
            'range_start': -1,
            'range_end': 1,
            'true_int': 0,
            'exp': 1
        }
    ]
    weight_dict = {'c1': 1, 'c2': 1, 'w': 1}
    error = False
    new_parameters, current_speeds = pt.prepare_new_day(
        personal_bests,
        parameter_dicts,
        best_parameters,
        current_speeds,
        value_dicts,
        weight_dict
    )
    assert new_parameters != None


def test_check_numeric():
    variables1 = [0, 9, 0.99, 'a']
    variables2 = [0.99, 1/3, 0, 100, 1e3]
    result1 = pt.check_numeric(variables1)
    result2 = pt.check_numeric(variables2)
    assert result1
    assert not result2