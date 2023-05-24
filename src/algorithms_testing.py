import gc
import os
import time

import numpy as np

from src import utilities, GreedyAlgorithm, AntColony
from src.data_generator import generate_compatibility_matrix

import branchAndBound as bnb
def main():
    # algo needs
    # number of elements from each of three groups
    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }
    max_samples_for_branch = 1
    start_level = 0

    number_of_iterations = 15
    ants_per_edge = 2
    influence_data = 1
    influence_pheromone = 0.8
    evaporation_coef = 0.1
    seed_start_point = 0

    # testing bnb orig
    execution_time_bnb_orig = []
    bnb_orig_results_json = []
    distributions = ['normalvariate', 'lognormvariate', 'standard']
    for method in distributions:
        for i in range(10):
            data = generate_compatibility_matrix(18, 18, 18, method)

            group_indices = utilities.get_group_indices(data)

            start_time = time.process_time()

            high_bound = bnb.high_bound(data, group_indices, how_much_to_choose)

            tr_tree = bnb.branch_and_bound(
                data,
                group_indices,
                how_much_to_choose,
                max_samples_for_branch,
                start_level)

            dict_for_json = {}

            for level in tr_tree:
                dict_for_json.update(level)

            final_tree, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

            end_time = time.process_time() - start_time

            execution_time_bnb_orig.append(end_time)
            bnb_orig_results_json.append(dict_for_json)

            gc.collect()

    # testing ant colony
    execution_time_ant = []
    ants_result = []
    for i in range(10):
        data = generate_compatibility_matrix(4 + i * 2, 4 + i * 2, 4 + i * 2, '')
        group_indices = utilities.get_group_indices(data)
        np.random.seed(seed_start_point)

        group_indices = utilities.get_group_indices(data)
        pheromone = np.ones(data.shape)

        start_time = time.process_time()

        GreedyAlgorithm.greedy_value(data, group_indices, how_much_to_choose)

        total_max_ant = {}
        total_corr_value = 0

        traversed_edges = []
        popular_vertices = []

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

            for key, values in iter_max_ant.items():
                if values['value'] > total_corr_value:
                    total_max_ant = {key: values}
                    total_corr_value = values['value']

            traversed_edges.append({})
            popular_vertices.append({})
            for key in ants_paths[-1].keys():
                edge_tuples = AntColony.traversed_edges_from_string(
                    group_indices, key)
                for tuple in edge_tuples:
                    if tuple in traversed_edges[-1]:
                        traversed_edges[-1][tuple] += 1
                    elif (tuple[::-1]) in traversed_edges[-1]:
                        traversed_edges[-1][tuple[::-1]] += 1
                    else:
                        traversed_edges[-1].update({tuple: 1})
                for tuple in edge_tuples:
                    if tuple[1] in popular_vertices[-1]:
                        popular_vertices[-1][tuple[1]] += 1
                    else:
                        popular_vertices[-1].update({tuple[1]: 0})

        ants_paths_and_edges = AntColony.add_edge_traverses_to_strings(
            ants_paths, traversed_edges[-1], group_indices)

        ants_tree_dict, _ = bnb.data_dict_to_treedict_converter(
            ants_paths_and_edges)

        execution_time_ant.append(time.process_time() - start_time)
        ants_result.append(ants_tree_dict)

    # testing bnb orig
    execution_time_bnb_modified = []
    bnb_modified_results_json = []
    data = generate_compatibility_matrix(18, 18, 18)
    for start_level_iter in range(1, 6):
        for max_samples_for_branch_iter in range(1, 10):
            group_indices = utilities.get_group_indices(data)

            start_time = time.process_time()

            high_bound = bnb.high_bound(data, group_indices, how_much_to_choose)

            tr_tree = bnb.branch_and_bound(
                data,
                group_indices,
                how_much_to_choose,
                max_samples_for_branch_iter,
                start_level_iter)

            dict_for_json = {}

            for level in tr_tree:
                dict_for_json.update(level)

            final_tree, max_dict = bnb.data_dict_to_treedict_converter(tr_tree)

            end_time = time.process_time() - start_time

            execution_time_bnb_modified.append(end_time)
            bnb_modified_results_json.append(dict_for_json)

            gc.collect()

    kek = 5

if __name__ == "__main__":
    main()