from django.core.management.base import BaseCommand, CommandError

import datetime
import math
import matplotlib.pyplot as plt
from mediation import learning
import os
import pytz
import rapidjson
import re
import sys

class Command(BaseCommand):
    help = "Runs the learning model case study"

    def add_arguments(self, parser):
        parser.add_argument("--plot_dir", type=str, default="")
        parser.add_argument(
            "args", 
            nargs="*", 
            help="Parameters for learning model: input_noise::float output_noise::float replications::int",
        )
        return

    def handle(self, *args, **options):
        if len(args) == 3:
            self.run_case_study(float(args[0]), float(args[1]), int(args[2]))
        elif options["plot_dir"] != "":
            self.plot_case_study(options["plot_dir"])
        return

    def run_case_study(self, input_noise, output_noise, replication):
        PARENT_DIR = "."
        model = learning.LearningModel(
            weather_file=PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
            plant_design_path=PARENT_DIR + "/config/plant_design.json",
            mediator_params_path=PARENT_DIR + "/config/mediator_settings.json",
            dispatch_params_path=PARENT_DIR + "/config/dispatch_settings.json",
            datetime_start=datetime.datetime(2021, 6, 1, 0, 0, 0, tzinfo=pytz.UTC),
            datetime_end=datetime.datetime(2021, 6, 3, 0, 0, 0, tzinfo=pytz.UTC),
            outputs=["m_dot_rec", "T_rec_out"],
        )
        model.run_case_study(
            input_noise=input_noise,
            output_noise=output_noise,
            replication=replication,
        )
        return

    def plot_case_study(self, dir):
        sol = Solutions(dir)
        sol.plot_ground_truth('m_dot_rec')
        sol.plot_ground_truth('T_rec_out')
        sol.plot_primal_trajectories('m_dot_rec', 0.02, 0.02)
        sol.plot_primal_trajectories('T_rec_out', 0.02, 0.02)
        sol.plot_boxplots()
        sol.plot_loss_trajectories()
        for i in range(10):
            sol.plot_scatter(i)
        sol.plot_scatter_2(0.02, 0.02)
        return

class SolutionRun:
    def __init__(self, filename):
        with open(filename, "r") as io:
            lines = io.readlines()
            self.evaluations = [rapidjson.loads(l) for l in lines]
            inputs = self._get_inputs(filename)
            self.input_noise = float(inputs[0])
            self.output_noise = float(inputs[1])
            self.replication = int(inputs[2])
        with open('%s_truth.json' % filename) as io:
            self.ground_truth = rapidjson.load(io)
        with open('%s_best_guess.json' % filename) as io:
            self.best_guess = rapidjson.load(io)
        return

    def _get_inputs(self, filename):
        m = re.match(".*?([0-9\.]+)\_([0-9\.]+)\_([0-9]+)\.jsonl", filename)
        return m[1], m[2], m[3]

    def get_mae(self, key):
        err, n = 0.0, 0
        for (xi, yi) in zip(self.ground_truth[key], self.best_guess[key]):
            if xi > 0.0 or yi > 0.0:
                n += 1
                err += abs(xi - yi) / max(xi, yi)
        return err / n

    def plot_primal_trajectory(self, ax, key):
        ax.set_title(
            'in, out = %s, %s' % (self.input_noise, self.output_noise),
            fontsize=10,
        )
        ax.plot(self.ground_truth[key], color='red', linewidth=0.5)
        ax.plot(self.best_guess[key], color='black', linewidth=0.5)
        return

    def plot_scatter(self, ax):
        x = [x['x']['avg_price_disp_storage_incentive'] for x in self.evaluations]
        y = [x['x']['helio_optical_error_mrad'] for x in self.evaluations]
        z = [x['f'] for x in self.evaluations]
        im = ax.scatter(x,y, c=z, cmap='viridis', alpha=0.3)
        ax.set_title(
            'in, out = %s, %s' % (self.input_noise, self.output_noise),
            fontsize=10,
        )
        sol = self.get_best('x')
        ax.plot(
            [sol['avg_price_disp_storage_incentive']],
            [sol['helio_optical_error_mrad']],
            'ko ',
            color = 'red'
        )
        ax.plot([0.0], [1.53], 'k* ', color='red')
        return

    def get_best(self, key = 'x'):
        best_f, best_x = 1e60, None
        for x in self.evaluations:
            if x['f'] < best_f:
                best_f = x['f']
                best_x = x['x']
        return best_x if key == 'x' else best_f

    def get_best_trajectory(self):
        x, y = [0], [self.evaluations[0]['f']]
        for i in range(len(self.evaluations)):
            f = self.evaluations[i]['f']
            if f < y[-1]:
                x.append(i)
                y.append(f)
        x.append(len(self.evaluations))
        y.append(y[-1])
        return x, y

    def plot_loss_trajectory(self, ax):
        x, y = self.get_best_trajectory()
        ax.set_title(
            'in, out = %s, %s' % (self.input_noise, self.output_noise),
            fontsize=10,
        )
        ax.plot(x, y, color='purple', alpha=0.5)
        ax.set_ylim(0, 50)
        return

class Solutions:
    def __init__(self, dir):
        self.dir = dir
        self.files = self._get_files(dir)
        self.input_noise = list(set([k[0] for k in self.files.keys()]))
        self.output_noise = list(set([k[1] for k in self.files.keys()]))
        self.input_noise.sort()
        self.output_noise.sort()
        return
 
    def _get_files(self, dir):
        files = {}
        filenames = os.listdir(dir)
        filenames.sort()
        for file in filenames:
            if not file.endswith('.jsonl'):
                continue
            sol = SolutionRun(os.path.join(dir, file))
            key = (sol.input_noise, sol.output_noise)
            if key not in files:
                files[key] = []
            files[key].append(sol)
        return files

    def _plot_boxplots(self, key, default):
        errors = []
        xlabels = []
        for kinput in self.input_noise:
            for koutput in self.output_noise:
                errors.append([
                    sol.get_best('x')[key]
                    for sol in self.files[(kinput, koutput)]
                ])
                xlabels.append('(%s, %s)' % (kinput, koutput))
        fig, ax = plt.subplots()
        ax.boxplot(errors, labels=xlabels)
        ax.axhline(y=default, color='red')
        ax.set_xlabel("(input, output) noise")
        ax.set_ylabel("Estimate of %s" % key)
        plt.savefig(os.path.join(self.dir, 'boxplot_%s.pdf' % key))
        return

    def _plot_error_boxplots(self):
        errors = []
        xlabels = []
        for kinput in self.input_noise:
            for koutput in self.output_noise:
                errors.append([
                    sol.get_best('f')
                    for sol in self.files[(kinput, koutput)]
                ])
                xlabels.append('(%s, %s)' % (kinput, koutput))
        fig, ax = plt.subplots()
        ax.boxplot(errors, labels=xlabels)
        ax.set_xlabel("(input, output) noise")
        ax.set_ylabel("Loss")
        plt.savefig(os.path.join(self.dir, 'boxplot_loss.pdf'))
        return

    def _plot_rmse_boxplots(self, key):
        errors = []
        xlabels = []
        for kinput in self.input_noise:
            for koutput in self.output_noise:
                errors.append([
                    100 * sol.get_mae(key) for sol in self.files[(kinput, koutput)]
                ])
                xlabels.append('(%s, %s)' % (kinput, koutput))
        fig, ax = plt.subplots()
        ax.boxplot(errors, labels=xlabels)
        ax.set_xlabel("(input, output) noise")
        ax.set_ylabel("Mean Absolute Percentage Error")
        plt.savefig(os.path.join(self.dir, 'boxplot_%s.pdf' % key))
        return

    def plot_boxplots(self):
        self._plot_boxplots('helio_optical_error_mrad', 1.53)
        self._plot_boxplots('avg_price_disp_storage_incentive', 0.0)
        self._plot_error_boxplots()
        self._plot_rmse_boxplots('T_rec_out')
        self._plot_rmse_boxplots('m_dot_rec')
        return

    def plot_loss_trajectories(self):
        fig, ax = plt.subplots(
            len(self.input_noise),
            len(self.output_noise),
            sharex = True,
            sharey = True,
        )
        for (i, kinput) in enumerate(self.input_noise):
            for (j, koutput) in enumerate(self.output_noise):
                for sol in self.files[(kinput, koutput)]:
                    sol.plot_loss_trajectory(ax[i][j])
        ax[-1][1].set_xlabel('Iteration')
        ax[1][0].set_ylabel('Loss')
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.4, 
            hspace=0.4,
        )
        plt.savefig(os.path.join(self.dir, 'trajectories.pdf'))
        plt.close(fig)
        return

    def plot_primal_trajectories(self, key, kinput, koutput):
        fig, ax = plt.subplots(
            3, 
            3,
            sharex = True,
            sharey = True,
        )
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                sol = self.files[(kinput, koutput)][k]
                sol.plot_primal_trajectory(ax[i][j], key)
                ax[i][j].set_title('Replication %i' % (k + 1))
        ax[-1][1].set_xlabel('Time')
        ax[1][0].set_ylabel(key)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.4, 
            hspace=0.4,
        )
        plt.savefig(os.path.join(self.dir, '%s.pdf' % key))
        plt.close(fig)
        return

    def plot_ground_truth(self, key):
        fig, ax = plt.subplots(
            len(self.input_noise),
            len(self.output_noise),
            sharex = True,
            sharey = True,
        )
        truth = self.files[(0.0, 0.0)][0]
        for (i, kinput) in enumerate(self.input_noise):
            for (j, koutput) in enumerate(self.output_noise):
                for sol in self.files[(kinput, koutput)]:
                    ax[i][j].plot(
                        sol.ground_truth[key],
                        linewidth = 0.5,
                        alpha = 0.5,
                        color = 'gray',
                    )
                    ax[i][j].set_title(
                        'in, out = %s, %s' % (kinput, koutput),
                        fontsize=10,
                    )
                    ax[i][j].plot(
                        truth.ground_truth[key],
                        linewidth = 0.5,
                        color = 'red',
                    )
        ax[-1][1].set_xlabel('Time')
        ax[1][0].set_ylabel(key)
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4,
        )
        plt.savefig(
            os.path.join(self.dir, 'ground_truth_%s.pdf' % key),
        )
        plt.close(fig)
        return

    def plot_scatter(self, replication = 0):
        fig, ax = plt.subplots(
            len(self.input_noise),
            len(self.output_noise),
            sharex = True,
            sharey = True,
        )
        for (i, kinput) in enumerate(self.input_noise):
            for (j, koutput) in enumerate(self.output_noise):
                k = 0 if i+j == 0 else replication
                sol = self.files[(kinput, koutput)][k]
                sol.plot_scatter(ax[i][j])
        ax[-1][1].set_xlabel('avg_price_disp_storage_incentive')
        ax[1][0].set_ylabel('helio_optical_error_mrad')
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.4,
            hspace=0.4,
        )
        plt.savefig(
            os.path.join(self.dir, 'sampled_points_%s.pdf' % replication),
        )
        plt.close(fig)
        return

    def plot_scatter_2(self, input_noise, output_noise):
        fig, ax = plt.subplots(3, 3, sharex = True, sharey = True)
        for i in range(3):
            for j in range(3):
                sol = self.files[(input_noise, output_noise)][3 * i + j]
                sol.plot_scatter(ax[i][j])
                ax[i][j].plot(
                    [0.0], [1.53], 'k* ', color='red',
                )
                ax[i][j].set_title('Replication %s' % (3 * i + j + 1))
        ax[-1][1].set_xlabel('avg_price_disp_storage_incentive')
        ax[1][0].set_ylabel('helio_optical_error_mrad')
        plt.subplots_adjust(
            left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.4, 
            hspace=0.4,
        )
        plt.savefig(
            os.path.join(self.dir, 'sampled_points_%s_%s.pdf' % (input_noise, output_noise)),
        )
        plt.close(fig)
        return
