# Tests for the learning.py module. They can be run with:
#  $ python -mpytest tests/learning.py

import pytest

from django.conf import settings
@pytest.fixture(scope='session')
def django_db_setup():
    settings.DATABASES['default'] = {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db.sqlite3',
    }
    return

import datetime
from mediation import learning
import os
import pathlib
import pytz
import rapidjson

PARENT_DIR = str(pathlib.Path(__file__).parents[1])

def test_local_solve():
    number_of_function_calls = 0
    def black_box_function(**params):
        nonlocal number_of_function_calls
        number_of_function_calls += 1
        return sum(v**2 for v in params.values())
    alg = learning.LearningAlgorithm(
        black_box_function,
        parameter_bounds={'x': (1, 2), 'y': (-2, 1)},
        global_search_iteration_limit=0,
        local_search_iteration_limit=20
    )
    f, x = alg.solve()
    assert(abs(f - 1) < 1e-6)
    assert(abs(x['x'] - 1) < 1e-6)
    assert(abs(x['y'] - 0) < 1e-6)
    assert(number_of_function_calls == 20)
    return

def test_global_solve():
    number_of_function_calls = 0
    def black_box_function(**params):
        nonlocal number_of_function_calls
        number_of_function_calls += 1
        return sum(v**2 for v in params.values())
    alg = learning.LearningAlgorithm(
        black_box_function,
        parameter_bounds={'x': (1, 2), 'y': (-2, 1)},
        global_search_iteration_limit=30,
        local_search_iteration_limit=0
    )
    f, x = alg.solve()
    assert(abs(f - 1) < 1e-1)
    assert(abs(x['x'] - 1) < 1e-1)
    assert(abs(x['y'] - 0) < 1e-1)
    assert(number_of_function_calls == 31)
    return

def test_global_local_solve():
    number_of_function_calls = 0
    def black_box_function(**params):
        nonlocal number_of_function_calls
        number_of_function_calls += 1
        return sum(v**2 for v in params.values())
    alg = learning.LearningAlgorithm(
        black_box_function,
        parameter_bounds={'x': (1, 2), 'y': (-2, 1)},
        global_search_iteration_limit=5,
        local_search_iteration_limit=10,
        log_filename=PARENT_DIR + "/test.jsonl"
    )
    f, x = alg.solve()
    assert(abs(f - 1) < 1e-2)
    assert(abs(x['x'] - 1) < 1e-1)
    assert(abs(x['y'] - 0) < 1e-1)
    # Because we have some initial points as well.
    assert(number_of_function_calls == 60)
    with open(PARENT_DIR + "/test.jsonl", "r") as io:
        lines = io.readlines()
        assert(len(lines) == 60)
        found_sol = False
        for l in lines:
            line = rapidjson.loads(l) 
            if line["f"] == f and line["x"] == x:
                found_sol = True
        assert(found_sol)
    os.remove(PARENT_DIR + "/test.jsonl")
    return

@pytest.mark.django_db
def test_evaluate():
    model = learning.LearningModel(
        weather_file=PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
        plant_design_path=PARENT_DIR + "/config/plant_design.json",
        mediator_params_path=PARENT_DIR + "/config/mediator_settings.json",
        dispatch_params_path=PARENT_DIR + "/config/dispatch_settings.json",
        datetime_start=datetime.datetime(2021, 6, 1, 0, 0, 0, tzinfo=pytz.UTC),
        datetime_end=datetime.datetime(2021, 6, 3, 0, 0, 0, tzinfo=pytz.UTC),
        outputs=['m_dot_rec', 'T_rec_out'],
    )
    ground_truth = model.noisy_evaluate(
        input_noise=0.05, 
        output_noise=0.05,
    )
    assert(len(ground_truth) == 2)
    assert(len(ground_truth['m_dot_rec']) == 48 * 12)
    assert(len(ground_truth['T_rec_out']) == 48 * 12)
    black_box_function = model.construct_objective(ground_truth)
    loss = black_box_function(
        helio_optical_error_mrad=1.53,
        avg_price_disp_storage_incentive=0.0,
    )
    assert(loss > 0)
    return

def test_global_solve_voronoi():
    number_of_function_calls = 0
    def black_box_function(**params):
        nonlocal number_of_function_calls
        number_of_function_calls += 1
        return sum(v**2 for v in params.values())
    alg = learning.LearningAlgorithm(
        black_box_function,
        parameter_bounds={'x': (1, 2), 'y': (-2, 1)},
        global_search_iteration_limit=30,
        local_search_iteration_limit=0,
        global_algorithm = learning.VoronoiSearch(),
    )
    f, x = alg.solve()
    assert(abs(f - 1) < 1e-1)
    assert(abs(x['x'] - 1) < 1e-1)
    assert(abs(x['y'] - 0) < 1e-1)
    return
