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

def average(
        array: list):
    return sum(array) / len(array)

def bnb_mean_testing(
        task_size: int,
        max_samples_for_branch: int,
        start_level,
        mean: float,
        mean_step: float,
        iterations: int,
        output_path: string = ''):
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]

    save_to_file = False
    if task_size == -1:
        task_size = 8
    if max_samples_for_branch == -1:
        max_samples_for_branch = 3
    if start_level == -1:
        start_level = 0
    if mean == -1:
        mean = 0.5
    if mean_step == -1:
        mean_step = 0.1
    if iterations == -1:
        iterations = 10
    if output_path != '':
        save_to_file = True

    max_result_bnb_orig = []
    execution_time_bnb_orig = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    max_result_bnb_orig_iteration = []
    execution_time_bnb_orig_iteration = []
    for j in np.arange(mean, 0.9, mean_step):
        for i in range(iterations):
            data = generate_compatibility_matrix(
                task_size,
                task_size,
                task_size,
                mean=j,
                dispersion=1)

            group_indices = utilities.get_group_indices(data)

            print("reg bnb iteration ", i+1, " of ", iterations)

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
        execution_time_bnb_orig.append(average(execution_time_bnb_orig_iteration))

        max_result_bnb_modified_iteration = []
        execution_time_bnb_modified_iteration = []
        # gc.collect()
        for i in range(iterations):
            print("bnb modified iteration ", i+1, " of ", iterations)
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
                    max_samples_for_branch,
                    0)

                _, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

                algo_results = next(iter(max_dict.values()))['value']
                if current_max < algo_results:
                    current_max = algo_results
            end_time = time.process_time() - start_time

            max_result_bnb_modified_iteration.append(current_max)
            execution_time_bnb_modified_iteration.append(end_time)

        max_result_bnb_modified.append(average(max_result_bnb_modified_iteration))
        execution_time_bnb_modified.append(average(execution_time_bnb_modified_iteration))
    value_to_stop_debugger = 'hello'

def bnb_dispersion_testing(
        task_size: int,
        max_samples_for_branch: int,
        start_level,
        dispersion: float,
        dispersion_step: float,
        iterations: int,
        output_path: string = ''):
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]

    save_to_file = False
    if task_size == -1:
        task_size = 8
    if max_samples_for_branch == -1:
        max_samples_for_branch = 3
    if start_level == -1:
        start_level = 0
    if dispersion == -1:
        dispersion = 0.5
    if dispersion_step == -1:
        dispersion_step = 0.1
    if iterations == -1:
        iterations = 10
    if output_path != '':
        save_to_file = True

    max_result_bnb_orig = []
    execution_time_bnb_orig = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    max_result_bnb_orig_iteration = []
    execution_time_bnb_orig_iteration = []

    for j in np.arange(dispersion, 5, dispersion_step):
        for i in range(iterations):
            data = generate_compatibility_matrix(
                task_size,
                task_size,
                task_size,
                mean=0.5,
                dispersion=j)

            group_indices = utilities.get_group_indices(data)

            print("reg bnb iteration ", i+1, " of ",  iterations)

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
        execution_time_bnb_orig.append(average(execution_time_bnb_orig_iteration))

        max_result_bnb_modified_iteration = []
        execution_time_bnb_modified_iteration = []
        # gc.collect()
        for i in range(iterations):
            print("bnb modified iteration ", i+1, " of ", iterations)
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
                    max_samples_for_branch,
                    start_level=0)

                _, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

                algo_results = next(iter(max_dict.values()))['value']
                if current_max < algo_results:
                    current_max = algo_results
            end_time = time.process_time() - start_time

            max_result_bnb_modified_iteration.append(current_max)
            execution_time_bnb_modified_iteration.append(end_time)

        max_result_bnb_modified.append(average(max_result_bnb_modified_iteration))
        execution_time_bnb_modified.append(average(execution_time_bnb_modified_iteration))
    stop_debugger_point = "stop me here"

def main():
    task_size = -1
    max_samples_for_branch = -1
    start_level = -1
    mean = -1
    mean_step = -1
    dispersion = -1
    dispersion_step = -1
    iterations = -1
    output_path = ''

    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["max_sfb=",
                                "task_size=",
                                "start_level=",
                                "mean=",
                                "mean_step=",
                                "dispersion=",
                                "dispersion_step=",
                                "iterations=",
                                "output_path="])

    for opt, arg in opts:
        if opt == '-h':
            print(
                'python bnb_testing.py [--max_sfb=<max samples for branch>]'
                ' [--task_size=<task size>] [--start_level=<step size>]'
                ' [--mean=<mean>] [--mean_step=<mean_step>] [--dispersion=<dispersion>]'
                ' [--dispersion_step=<dispersion step>] [--iterations=<iterations>]'
                ' [--output_path=<output_path>]')
            sys.exit()
        elif opt in "--max_sfb=":
            max_samples_for_branch = int(arg)
        elif opt in "--task_size=":
            task_size = int(arg)
        elif opt in "--start_level=":
            start_level = int(arg)
        elif opt in "--mean=":
            mean = float(arg)
        elif opt in "--mean_step=":
            mean_step = float(arg)
        elif opt in "--dispersion=":
            dispersion = float(arg)
        elif opt in "--dispersion_step=":
            dispersion_step = float(arg)
        elif opt in "--iterations=":
            iterations = int(arg)
        elif opt in "--output_path=":
            output_path = arg

    if max_samples_for_branch == '':
        max_samples_for_branch = -1
    if mean == '':
        mean = -1
    if task_size == '':
        task_size = -1
    if start_level == '':
        start_level = -1
    if mean_step == '':
        mean_step = -1
    if dispersion == '':
        dispersion = -1
    if dispersion_step == '':
        dispersion_step = -1
    if iterations == '':
        iterations = -1

    bnb_mean_testing(
        task_size,
        max_samples_for_branch,
        start_level,
        mean,
        mean_step,
        iterations,
        output_path)

    bnb_dispersion_testing(
        task_size,
        max_samples_for_branch,
        start_level,
        dispersion,
        dispersion_step,
        iterations,
        output_path)

if __name__ == "__main__":
    main()
