import gc
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

import utilities
import GreedyAlgorithm
import AntColony
from data_generator import generate_compatibility_matrix

import branchAndBound as bnb


def average(
        array: list):
    return sum(array) / len(array)


def OverallTimeTesting(min_task_size=4, max_task_size=14, step=1):
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
    for i in range(min_task_size, max_task_size, step):
        task_size.append(i)

    execution_time_bnb_orig = []
    max_result_bnb_orig = []

    execution_time_ant = []
    max_result_ant = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    mean = 0.5
    dispersion = 0.25

    # std = data.std()
    # mean = data.mean()
    # disperse = math.pow(std, 2)

    for size in task_size:
        max_result_bnb_orig_iteration = []
        execution_time_bnb_orig_iteration = []
        for i in range(10):
            data = generate_compatibility_matrix(size, size, size, 'normalvariate', mean, dispersion)

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
        execution_time_bnb_orig.append(average(execution_time_bnb_orig_iteration))

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
            print("bnb modified iteration ", size, ".", i, " of ", max_task_size)
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

        max_result_bnb_modified.append(average(max_result_bnb_modified_iteration))
        execution_time_bnb_modified.append(average(execution_time_bnb_modified_iteration))

        gc.enable()

    print(execution_time_bnb_orig)
    print(max_result_bnb_orig)

    print(execution_time_ant)
    print(max_result_ant)

    print(execution_time_bnb_modified)
    print(max_result_bnb_modified)

    def PlotData():
        fig1, ax1 = plt.subplots()
        ax1.plot(task_size, execution_time_bnb_orig,
                 "-b<", label="branching tree")
        ax1.plot(task_size, execution_time_ant, "-g>", label="ant colony")
        ax1.plot(task_size, execution_time_bnb_modified,
                 "-rh", label="modified br tree")
        plt.legend(loc="upper left")
        plt.title(
            "Залежність часу виконання від розмірності")
        plt.xlabel("розмір групи")
        plt.ylabel("час роботи, с")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.plot(task_size, max_result_bnb_orig,
                 "-b>", label="branching tree")
        ax2.plot(task_size, max_result_ant, "-g<", label="ant colony")
        ax2.plot(task_size, max_result_bnb_modified,
                 "-rh", label="modified br tree")
        plt.legend(loc="upper left")
        plt.title(
            "Залежність точності від розмірності")
        plt.xlabel("розмір групи")
        plt.ylabel("загальна взаємопридатність, од")

        plt.ylim(0,15)

        plt.show()

    PlotData()


def bnb_mean_testing(
        task_size: int,
        max_samples_for_branch: int,
        start_level,
        mean: float,
        mean_step: float,
        iterations: int = 10):
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]

    max_result_bnb_orig = []
    execution_time_bnb_orig = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    max_result_bnb_orig_iteration = []
    execution_time_bnb_orig_iteration = []

    for j in range(mean, 0.9, mean_step):
        for i in range(iterations):
            data = generate_compatibility_matrix(
                task_size,
                task_size,
                task_size,
                mean=j,
                dispersion=1)

            group_indices = utilities.get_group_indices(data)

            print("reg bnb iteration ", i, " of ", task_size)

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
            print("bnb modified iteration ", i, " of ", task_size)
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
        iterations: int = 10):
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]

    max_result_bnb_orig = []
    execution_time_bnb_orig = []

    execution_time_bnb_modified = []
    max_result_bnb_modified = []

    max_result_bnb_orig_iteration = []
    execution_time_bnb_orig_iteration = []

    for j in range(dispersion, 5, dispersion_step):
        for i in range(iterations):
            data = generate_compatibility_matrix(
                task_size,
                task_size,
                task_size,
                mean=0.5,
                dispersion=j)

            group_indices = utilities.get_group_indices(data)

            print("reg bnb iteration ", i, " of ", task_size)

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
            print("bnb modified iteration ", i, " of ", task_size)
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
                    start_level)

                _, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

                algo_results = next(iter(max_dict.values()))['value']
                if current_max < algo_results:
                    current_max = algo_results
            end_time = time.process_time() - start_time

            max_result_bnb_modified_iteration.append(current_max)
            execution_time_bnb_modified_iteration.append(end_time)

        max_result_bnb_modified.append(average(max_result_bnb_modified_iteration))
        execution_time_bnb_modified.append(average(execution_time_bnb_modified_iteration))
    k = 5

def main():
    # algo needs
    # number of elements from each of three groups
    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }
    all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]
    depth = range(500)
    allData = [[x, 2 * x, 3 * x] for x in depth]
    max_samples_for_branch = 1
    start_level = 0

    years = [6, 8, 10, 12, 15]
    first = [2, 3, 4, 5, 6]
    second = [3, 4, 6, 5, 4]

    # OverallTimeTesting()
    bnb_mean_testing(7, 2, 2, 0.5, 0.1, 10)
    bnb_dispersion_testing(7, 2, 2, 0.5, 0.1, 10)

    # number_of_iterations = 15
    # ants_per_edge = 2
    # influence_data = 1
    # influence_pheromone = 0.8
    # evaporation_coef = 0.1
    # seed_start_point = 0

    # # testing bnb orig
    # execution_time_bnb_orig = []
    # bnb_orig_results_json = []
    # distributions = ['normalvariate', 'lognormvariate', 'standard']
    # for method in distributions:
    #     for i in range(10):
    #         data = generate_compatibility_matrix(18, 18, 18, method)

    #         group_indices = utilities.get_group_indices(data)

    #         start_time = time.process_time()

    #         high_bound = bnb.high_bound(
    #             data, group_indices, how_much_to_choose)

    #         tr_tree = bnb.branch_and_bound(
    #             data,
    #             group_indices,
    #             how_much_to_choose,
    #             max_samples_for_branch,
    #             start_level)

    #         dict_for_json = {}

    #         for level in tr_tree:
    #             dict_for_json.update(level)

    #         final_tree, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

    #         end_time = time.process_time() - start_time

    #         execution_time_bnb_orig.append(end_time)
    #         bnb_orig_results_json.append(dict_for_json)

    #         gc.collect()

    # # testing ant colony
    # execution_time_ant = []
    # ants_result = []
    # for i in range(10):
    #     data = generate_compatibility_matrix(
    #         4 + i * 2, 4 + i * 2, 4 + i * 2, '')
    #     group_indices = utilities.get_group_indices(data)
    #     np.random.seed(seed_start_point)

    #     group_indices = utilities.get_group_indices(data)
    #     pheromone = np.ones(data.shape)

    #     start_time = time.process_time()

    #     GreedyAlgorithm.greedy_value(data, group_indices, how_much_to_choose)

    #     total_max_ant = {}
    #     total_corr_value = 0

    #     traversed_edges = []
    #     popular_vertices = []

    #     for i in range(number_of_iterations):
    #         ants_paths = AntColony.ant_iteration(
    #             data, group_indices, how_much_to_choose, pheromone,
    #             ants_per_edge, influence_data, influence_pheromone)

    #         AntColony.update_pheromone_array(
    #             data, group_indices, how_much_to_choose, pheromone,
    #             ants_paths, evaporation_coef)

    #         for ant, values in ants_paths[-1].items():
    #             values['value'] = bnb.bound_from_string(
    #                 data, group_indices, how_much_to_choose, ant)

    #         # find max ant, but discard a tree because only last tree level is used
    #         _, iter_max_ant = bnb.data_dict_to_treedict_converter(
    #             [ants_paths[-1]])

    #         for key, values in iter_max_ant.items():
    #             if values['value'] > total_corr_value:
    #                 total_max_ant = {key: values}
    #                 total_corr_value = values['value']

    #         traversed_edges.append({})
    #         popular_vertices.append({})
    #         for key in ants_paths[-1].keys():
    #             edge_tuples = AntColony.traversed_edges_from_string(
    #                 group_indices, key)
    #             for tuple in edge_tuples:
    #                 if tuple in traversed_edges[-1]:
    #                     traversed_edges[-1][tuple] += 1
    #                 elif (tuple[::-1]) in traversed_edges[-1]:
    #                     traversed_edges[-1][tuple[::-1]] += 1
    #                 else:
    #                     traversed_edges[-1].update({tuple: 1})
    #             for tuple in edge_tuples:
    #                 if tuple[1] in popular_vertices[-1]:
    #                     popular_vertices[-1][tuple[1]] += 1
    #                 else:
    #                     popular_vertices[-1].update({tuple[1]: 0})

    #     ants_paths_and_edges = AntColony.add_edge_traverses_to_strings(
    #         ants_paths, traversed_edges[-1], group_indices)

    #     ants_tree_dict, _ = bnb.data_dict_to_treedict_converter(
    #         ants_paths_and_edges)

    #     execution_time_ant.append(time.process_time() - start_time)
    #     ants_result.append(ants_tree_dict)

    # # testing bnb modified
    # execution_time_bnb_modified = []
    # bnb_modified_results_json = []
    # data = generate_compatibility_matrix(18, 18, 18)
    # for start_level_iter in range(1, 6):
    #     for max_samples_for_branch_iter in range(1, 10):
    #         group_indices = utilities.get_group_indices(data)

    #         start_time = time.process_time()

    #         high_bound = bnb.high_bound(
    #             data, group_indices, how_much_to_choose)

    #         tr_tree = bnb.branch_and_bound(
    #             data,
    #             group_indices,
    #             how_much_to_choose,
    #             max_samples_for_branch_iter,
    #             start_level_iter)

    #         dict_for_json = {}

    #         for level in tr_tree:
    #             dict_for_json.update(level)

    #         final_tree, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

    #         end_time = time.process_time() - start_time

    #         execution_time_bnb_modified.append(end_time)
    #         bnb_modified_results_json.append(dict_for_json)

    #         gc.collect()

    # kek = 5


if __name__ == "__main__":
    main()
