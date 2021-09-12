import bayes_opt
import datetime
import pytz
import math
from mediation import mediator
import numpy
import os
import pybobyqa
import rapidjson
import tempfile

class LearningAlgorithm:
    """
    A class to manage the learning algorithm. 

    The algorithm is a blend of global search using Bayesian Optimization and
    local search using BOBYQA. For each candidate solution found using Bayesian
    Optimization, we run local search to improve it to a local optima.

    To disable the inner local search, set `local_search_iteration_limit = 0`.
    To disable the outer Bayesian Optimization search and run pure local search,
    set `global_search_iteration_limit = 0`.

    Arguments
    ---------

     * `f`: the function to minimize. It takes the form`f(**params)` where
       `params` is a `dict` from string to value.
     * `global_search_iteration_limit`: the number of Bayesian Optimization
       major iterations to run.
     * `local_search_iteration_limit`: the number of pybobyqa minor iterations
       to run within each major Bayesian iteration.
     * `global_params`: a dictionary of keyword arguments passed to
       BayesianOptimization.maximize. See
       https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation_vs_exploration.ipynb
       for details.
     * `local_params`: a dictionary of keyword arguments passed to the
       `user_params` argument of pybobyqa.solve. See
       https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/advanced.html
       for details.

    Example
    -------

    ```python
    def f(**params):
        return (params['x'] - 1)**2
    alg = LearningAlgorithm(
        f, 
        {'x': (-1, 1)},
        global_search_iteration_limit = 4,
        local_search_iteration_limit = 4,
    )
    alg.solve()
    ```
    """
    def __init__(
        self, 
        f, 
        parameter_bounds, 
        global_search_iteration_limit,
        local_search_iteration_limit,
        global_params={
            # The number of initial points. These don't count as part of the
            # iteration limit.
            'init_points': 0,
            # The acquisition function is "expected improvement"
            'acq': 'ei',
            # Controls the exploration/exploitation with the expected
            # improvement objective function. Bigger xi => more exploration.
            'xi': 1e-2,
        },
        local_params={},
        log_filename = None,
    ):
        self.f = f
        self.parameter_bounds = parameter_bounds
        # Create an ordering for the parameters!
        self.parameters = [k for k in parameter_bounds.keys()]
        self.global_search_iteration_limit = global_search_iteration_limit
        self.global_params = global_params
        self.local_search_iteration_limit = local_search_iteration_limit
        self.local_params = local_params
        self.log_filename = log_filename
        if log_filename is not None:
            with open(log_filename, 'w') as io:
                pass  # Create a blank file.
        return

    def _vector_to_dict(self, x):
        """
        pybobyqa takes in a numpy array, but our evaluation function requires a
        named dictionary. This function handles the conversion.
        """
        return {k: xi for (k, xi) in zip(self.parameters, x)}

    def _dict_to_vector(self, x):
        """
        pybobyqa takes in a numpy array, but our evaluation function requires a
        named dictionary. This function handles the conversion.
        """
        return numpy.array([x[k] for k in self.parameters])
    
    def _log_solution(self, f, x):
        if self.log_filename is not None:
            with open(self.log_filename, 'a') as io:
                rapidjson.dump({'f': f, 'x': x}, io)
                io.write('\n')
        return

    def _local_search(self, x0):
        """
        Run local search from the point `x0`. Returns `(f_star, x_star)`
        corresponding to the best solution found.
        """
        if self.local_search_iteration_limit == 0:
            # No iterations! Just return the point the user passed in.
            f = self.f(**x0)
            self._log_solution(f, x0)
            return f, x0
        def local_f(x):
            x_dict = self._vector_to_dict(x)
            f = self.f(**x_dict)
            self._log_solution(f, x_dict)
            return f
        solution = pybobyqa.solve(
            local_f,
            self._dict_to_vector(x0),
            bounds=(
                self._dict_to_vector(
                    {k: v[0] for (k, v) in self.parameter_bounds.items()},
                ),
                self._dict_to_vector(
                    {k: v[1] for (k, v) in self.parameter_bounds.items()},
                ),
            ),
            scaling_within_bounds=True,
            user_params=self.local_params,
            maxfun=self.local_search_iteration_limit,
            objfun_has_noise=False,
            seek_global_minimum=True,
        )
        return solution.f, self._vector_to_dict(solution.x)

    def _global_search(self):
        """
        Run global search. Returns `(f_star, x_star)` corresponding to the best
        solution found.
        """
        # Hack-time: global search gets passed an `x`, but then forwards that to
        # the local search. So we need to maintain a mapping from input `x` to
        # output `x`. But Python doesn't like dictionaries as keys (because
        # they aren't hashable?), so we just use a list of input-output tuples.
        # It's slow, but not the bottleneck so we don't care.
        global_to_local_map = []
        def bayes_f(**x):
            f_star, x_star = self._local_search(x0=x)
            global_to_local_map.append((x.copy(), x_star))
            return -f_star
        optimizer = bayes_opt.BayesianOptimization(
            f=bayes_f,
            pbounds=self.parameter_bounds,
            # `random_state` is a seed for the random number generator. I guess
            # it's okay being `1`?
            random_state=1,
        )
        optimizer.maximize(
            n_iter=self.global_search_iteration_limit,
            **self.global_params,
        )
        for (global_param, local_param) in global_to_local_map:
            if global_param == optimizer.max['params']:
                return -optimizer.max['target'], local_param
        raise Exception("Something went wrong. Couldn't find the local solution")

    def solve(self):
        "Run the optimization!"
        if self.global_search_iteration_limit == 0:
            # If there are no global iterations, just use local search and don't 
            # use bayesian optimization. For simplicity, start the search from
            # the middle of the bounds.
            return self._local_search(x0={
                k: (v[0] + v[1]) / 2 for (k, v) in self.parameter_bounds.items()
            })
        return self._global_search()

class LearningModel:
    """
    A class to manage the training function for LORE.

    Arguments
    ---------

     * weather_file: filename of a TMY weather file
     * plant_design_path: filename of a JSON file for the plant config
     * mediator_params_path: filename of a JSON file for the mediator config
     * dispatch_params_path: filename of a JSON file for the dispatch config
     * datetime_start: UTC datetime for the start of the simulation
     * datetime_end: UTC datetime for the end of the simulation
     * outputs: a list of keys to return from the simulation
    """
    def __init__(
        self,
        weather_file,
        plant_design_path,
        mediator_params_path,
        dispatch_params_path,
        datetime_start,
        datetime_end,
        outputs,
    ):
        self.weather_file = weather_file
        self.plant_design_path = plant_design_path
        self.mediator_params_path = mediator_params_path
        self.dispatch_params_path = dispatch_params_path
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self.outputs = outputs
        return

    def evaluate(self, **params):
        """
        Evaluate the LORE simulation given the input parameters `params`.

        Returns a dictionary mapping output fields to a list of their values.
        """
        # Due to the way `mediator` is written, we need to read and write a new
        # config file with the changed values.
        tmp_plant_design_path = \
            self._replace_params(self.plant_design_path, params)
        tmp_mediator_params_path = \
            self._replace_params(self.mediator_params_path, params)
        m = mediator.Mediator(
            params_path=tmp_mediator_params_path,
            plant_design_path=tmp_plant_design_path,
            weather_file=self.weather_file,
            dispatch_params_path=self.dispatch_params_path,
            update_interval=datetime.timedelta(seconds=5),
        )
        ret = m.run_once(
            datetime_start=self.datetime_start,
            datetime_end=self.datetime_end,
            run_without_adding_to_database=True,
        )
        os.remove(tmp_plant_design_path)
        os.remove(tmp_mediator_params_path)
        return {k: ret[k] for k in self.outputs}

    def _replace_params(self, filename, params):
        with open(filename) as io:
            data = rapidjson.load(io, parse_mode=1)
            for (key, value) in params.items():
                if key in data:
                    data[key] = value
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as io:
            rapidjson.dump(data, io)
        return path

    def _root_mean_squared(self, x):
        return math.sqrt(sum(xk**2 for xk in x) / len(x))

    def _loss(self, target_solution, trial_solution):
        """
        The loss function between two solutions.
        """
        return {
            k: self._root_mean_squared([
                xk - yk for
                (xk, yk) in zip(target_solution[k], trial_solution[k])
            ])
            for k in target_solution.keys()
        }

    def _aggregate_solution(self, x, window):
        return {
            k: numpy.convolve(v, numpy.ones(window) / window, mode='valid')
            for (k, v) in x.items()
        }

    def construct_objective(self, target_solution, aggregate_window=12):
        """
        Given a target solution `target_solution`, which is a `dict` from key to
        a vector of values, return a function to be used in `LearningAlgorithm`
        that is the difference between to trial solution generated by the
        learning algorithm and the target solution.

        `aggregate_window` is a rolling window over which we average the values
        to remove any high-frequency swings.
        """
        aggregated_target_solution = \
            self._aggregate_solution(target_solution, aggregate_window)
        def black_box_function(**params):
            trial_solution = self.evaluate(**params)
            l = self._loss(
                aggregated_target_solution,
                self._aggregate_solution(trial_solution, aggregate_window),
            )
            return self._root_mean_squared(l.values())
        return black_box_function

    def _modify_weather_file(self, input_noise):
        fd, modified_weather_file = tempfile.mkstemp()
        def _scale_value(x):
            y = max(0, float(x) * numpy.random.normal(loc=1.0, scale=input_noise))
            return str(round(y))
        with open(self.weather_file, 'r') as io:
            with open(modified_weather_file, 'w') as out_io:
                row = 0
                for line in io:
                    row += 1
                    if row <= 3:  # First three rows of a TMY file are headers
                        out_io.write(line)
                    else:  # The meat of the TMY file
                        items = line.split(',')
                        # Index 5 is the DNI column. Scale that!
                        items[5] = _scale_value(items[5])
                        out_io.write(','.join(items))
        self.weather_file = modified_weather_file
        return

    def noisy_evaluate(self, input_noise, output_noise, **params):
        """
        Evaluate the model given the parameters `params` and two types of noise:

         * `input_noise` is standard deviation of a normal distribution with
           mean 1.0. We scale each element of input DNI by an independent
           realization of this random variable.
         * `output_noise` is standard deviation of a normal distribution with
           mean 1.0. We scale each element of output vectors by an independent
           realization of this random variable.

        To disable the noise, pass `input_noise=0` and/or `output_noise=0`.
        """
        # Scale the input by the noise
        if input_noise > 0:
            input_weather_file = self.weather_file
            self._modify_weather_file(input_noise)
        output = self.evaluate(**params)
        # Scale the output by the noise
        if output_noise > 0:
            for (k, v) in output.items():
                w = numpy.random.normal(loc=1.0, scale=output_noise, size=len(v))
                output[k] *= w
        # If we modified the weather file, reset the original version
        if input_noise > 0:
            os.remove(self.weather_file)
            self.weather_file = input_weather_file
        return output

    def run_case_study(
        self, 
        input_noise,
        output_noise,
        replication,
    ):
        """
        Run an experiment for the case-study.
        See `/mediation/management/commands/learning_case_study.py` for more
        details.
        """
        log_filename = \
            "%s_%s_%s.jsonl" % (input_noise, output_noise, replication)
        if os.path.isfile(log_filename):
            return
        ground_truth = self.noisy_evaluate(
            input_noise=input_noise,
            output_noise=output_noise,
        )
        with open("%s_truth.json" % log_filename, "w") as io:
            rapidjson.dump(
                {k: list(v) for (k, v) in ground_truth.items()},
                io,
            )
        black_box_function = self.construct_objective(ground_truth)
        alg = LearningAlgorithm(
            black_box_function,
            {
                'helio_optical_error_mrad': (1.0, 4.0),
                'avg_price_disp_storage_incentive': (0.0, 200.0),
            },
            global_search_iteration_limit=50,
            local_search_iteration_limit=0,
            log_filename=log_filename,
        )
        f, x = alg.solve()
        output = self.evaluate(**x)
        with open("%s_best_guess.json" % log_filename, "w") as io:
            rapidjson.dump(
                {k: list(v) for (k, v) in output.items()},
                io,
            )
        return
