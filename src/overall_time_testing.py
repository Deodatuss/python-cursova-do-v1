import gc
import getopt
import math
import os
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

import utilities, AntColony, branchAndBound as bnb
from data_generator import generate_compatibility_matrix
import GreedyAlgorithm


def average(
        array: list):
    return sum(array) / len(array)


def overall_time_testing(min_task_size=4, max_task_size=14, step=1, output_path: string = ''):
    save_to_file = False
    if min_task_size == -1:
        min_task_size = 4
    if max_task_size == -1:
        max_task_size = 14
    if step == -1:
        step = 1
    if output_path != '':
        save_to_file = True

    gc.disable()
    number_of_iterations = 15
    ants_per_edge = 1
    influence_data = 1
    influence_pheromone = 0.8
    evaporation_coef = 0.1
    seed_start_point = 0
    np.random.seed(seed_start_point)

    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]

    task_size = []
    for i in np.arange(min_task_size, max_task_size, step):
        task_size.append(round(i))

    execution_time_bnb_orig = []
    max_result_bnb_orig = []

    execution_time_ant = []
    max_result_ant = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    execution_time_greedy = []
    max_result_greedy = []

    mean = 0.5
    dispersion = 0.25

    # std = data.std()
    # mean = data.mean()
    # disperse = math.pow(std, 2)

    for size in task_size:
        max_result_bnb_orig_iteration = []
        execution_time_bnb_orig_iteration = []
        for i in range(10):
            data = generate_compatibility_matrix(
                size, size, size, 'normalvariate', mean, dispersion)

            group_indices = utilities.get_group_indices(data)

            print("reg bnb iteration ", size, ".", i, " of ", max_task_size)

            # gc.collect()
            start_time = time.process_time()

            current_max = -1
            for i in all_possible_choosings:
                how_much_to_choose = {
                    'a': i[0], 'b': i[1], 'c': i[2]
                }

                tr_tree = bnb.branch_and_bound(
                    data,
                    group_indices,
                    how_much_to_choose,
                    max_samples_for_branch=1,
                    start_level=0)

                _, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

                algo_results = next(iter(max_dict.values()))['value']
                if current_max < algo_results:
                    current_max = algo_results
            end_time = time.process_time() - start_time

            max_result_bnb_orig_iteration.append(current_max)
            execution_time_bnb_orig_iteration.append(end_time)

        max_result_bnb_orig.append(average(max_result_bnb_orig_iteration))
        execution_time_bnb_orig.append(
            average(execution_time_bnb_orig_iteration))

        # ant colony
        #
        execution_time_ant_iteration = []
        max_result_ant_iteration = []

        for i in range(10):
            print("ant colony iteration ", size, ".", i, " of ", max_task_size)

            current_max = -1
            total_iterations_time = 0
            start_time = time.process_time()

            for i in all_possible_choosings:
                # collect at every iteration, because ant colony is memory-costly
                # gc.collect()

                how_much_to_choose = {
                    'a': i[0], 'b': i[1], 'c': i[2]
                }
                pheromone = np.ones(data.shape)

                for i in range(number_of_iterations):
                    ants_paths = AntColony.ant_iteration(
                        data, group_indices, how_much_to_choose, pheromone,
                        ants_per_edge, influence_data, influence_pheromone)

                    AntColony.update_pheromone_array(
                        data, group_indices, how_much_to_choose, pheromone,
                        ants_paths, evaporation_coef)

                    for ant, values in ants_paths[-1].items():
                        values['value'] = bnb.bound_from_string(
                            data, group_indices, how_much_to_choose, ant)

                    # find max ant, but discard a tree because only last tree level is used
                    _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                        [ants_paths[-1]])
                    algo_results = next(iter(iter_max_ant.values()))['value']
                    if current_max < algo_results:
                        current_max = algo_results

                    # total_iterations_time += time.process_time() - start_time
            end_time = time.process_time() - start_time

            max_result_ant_iteration.append(current_max)
            execution_time_ant_iteration.append(end_time)

        max_result_ant.append(average(max_result_ant_iteration))
        execution_time_ant.append(average(execution_time_ant_iteration))

        # bnb modified
        #
        max_result_bnb_modified_iteration = []
        execution_time_bnb_modified_iteration = []
        # gc.collect()
        for i in range(10):
            print("bnb modified iteration ", size,
                  ".", i, " of ", max_task_size)
            start_time = time.process_time()

            current_max = -1
            for i in all_possible_choosings:
                how_much_to_choose = {
                    'a': i[0], 'b': i[1], 'c': i[2]
                }

                tr_tree = bnb.branch_and_bound(
                    data,
                    group_indices,
                    how_much_to_choose,
                    max_samples_for_branch=2,
                    start_level=1)

                _, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

                algo_results = next(iter(max_dict.values()))['value']
                if current_max < algo_results:
                    current_max = algo_results
            end_time = time.process_time() - start_time

            max_result_bnb_modified_iteration.append(current_max)
            execution_time_bnb_modified_iteration.append(end_time)

        max_result_bnb_modified.append(
            average(max_result_bnb_modified_iteration))
        execution_time_bnb_modified.append(
            average(execution_time_bnb_modified_iteration))

        # bnb modified
        #
        max_result_greedy_iteration = []
        execution_time_greedy_iteration = []
        # gc.collect()
        for i in range(10):
            print("greedy iteration ", size, ".", i, " of ", max_task_size)
            start_time = time.process_time()

            current_max = -1
            for i in all_possible_choosings:
                how_much_to_choose = {
                    'a': i[0], 'b': i[1], 'c': i[2]
                }

                value = GreedyAlgorithm.greedy_value(
                    data,
                    group_indices,
                    how_much_to_choose)

                if current_max < value:
                    current_max = value
            end_time = time.process_time() - start_time

            max_result_greedy_iteration.append(current_max)
            execution_time_greedy_iteration.append(end_time)
        max_result_greedy.append(
            average(max_result_greedy_iteration))
        execution_time_greedy.append(
            average(execution_time_greedy_iteration))

        gc.enable()

    print(execution_time_bnb_orig)
    print(max_result_bnb_orig)

    print(execution_time_ant)
    print(max_result_ant)

    print(execution_time_bnb_modified)
    print(max_result_bnb_modified)

    print(execution_time_greedy)
    print(max_result_greedy)

    def plot_data():
        fig1, ax1 = plt.subplots()
        ax1.plot(task_size, execution_time_bnb_orig,
                 "-b<", label="branching tree")
        ax1.plot(task_size, execution_time_ant, "-g>", label="ant colony")
        ax1.plot(task_size, execution_time_bnb_modified,
                 "-rh", label="modified br tree")
        ax1.plot(task_size, execution_time_greedy,
                 "-yd", label="greedy")
        plt.legend(loc="upper left")
        plt.title(
            "Залежність часу виконання від розмірності")
        plt.xlabel("розмір групи")
        plt.ylabel("час роботи, с")
        plt.show()

        if save_to_file:
            file_name = os.path.join(
                output_path, "Залежність часу виконання від розмірності.png")
            plt.savefig(file_name)

        fig2, ax2 = plt.subplots()
        ax2.plot(task_size, max_result_bnb_orig,
                 "-b>", label="branching tree")
        ax2.plot(task_size, max_result_ant, "-g<", label="ant colony")
        ax2.plot(task_size, max_result_bnb_modified,
                 "-rh", label="modified br tree")
        ax2.plot(task_size, max_result_greedy,
                 "-yd", label="greedy")
        plt.legend(loc="upper left")
        plt.title(
            "Залежність точності від розмірності")
        plt.xlabel("розмір групи")
        plt.ylabel("загальна взаємопридатність, од")

        plt.ylim(0, 15)

        plt.show()

        if save_to_file:
            file_name = os.path.join(
                output_path, "Залежність точності від розмірності.png")
            plt.savefig(file_name)

    plot_data()


def main():
    min_task_size = -1
    max_task_size = -1
    step_size = -1
    output_path = ''

    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["min_ts=",
                                "max_ts=",
                                "step_size=",
                                "output_path="])

    for opt, arg in opts:
        if opt == '-h':
            print(
                'python overall_time_testing.py [--min_ts=<min task size>]'
                ' [--max_ts=<max task size>] [--step_size=<step size>]'
                ' [--output_path=<output path>]')
            sys.exit()
        elif opt in "--min_ts=":
            min_task_size = int(arg)
        elif opt in "--max_ts=":
            max_task_size = int(arg)
        elif opt in "--step_size=":
            step_size = float(arg)
        elif opt in "--output_path=":
            output_path = arg

    if min_task_size == '':
        min_task_size = -1
    if max_task_size == '':
        max_task_size = -1
    if step_size == '':
        step_size = -1

    overall_time_testing(min_task_size, max_task_size, step_size, output_path)


if __name__ == "__main__":
    main()
