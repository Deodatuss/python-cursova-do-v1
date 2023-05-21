import getopt
import os
import sys

import numpy as np
import converters
import utilities
import GreedyAlgorithm
import AntColony
import branchAndBound as bnb

from contextlib import redirect_stdout


def main():
    """
    This function is used as a test and presentation for Ant Colony Algorithm functions and workflow
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))

    input_file = ''
    output_folder = ''

    # number of elements from each of three groups
    how_much_to_choose = {
        'a': '', 'b': '', 'c': ''
    }
    number_of_iterations = ''
    ants_per_edge = ''
    influence_data = ''
    influence_pheromone = ''
    evaporation_coef = ''
    seed_start_point = ''

    opts, args = getopt.getopt(sys.argv[1:], "hi:d:o:a:b:c:",
                               ["input_file=",
                                "output_path=",
                                "a=", "b=", "c=",
                                "id=",
                                "noi=",
                                "ape=",
                                "iph=",
                                "ecoef=",
                                "ssp="])

    is_default_values = next((arg for opt, arg in opts if opt == "-d" and arg == 'true'), None)

    if is_default_values is not None:
        input_file = os.path.join(root_folder, '..', 'data', 'demo', "input.json")
        output_folder = os.path.join(root_folder, '..', 'data', 'demo', 'ants')

        # number of elements from each of three groups
        how_much_to_choose = {
            'a': 1, 'b': 3, 'c': 2
        }
        number_of_iterations = 15
        ants_per_edge = 2
        influence_data = 1
        influence_pheromone = 0.8
        evaporation_coef = 0.1
        seed_start_point = 0
    else:
        for opt, arg in opts:
            if opt == '-h':
                print(
                    'python AntColonyDemo.py -i <inputfile.json> -o <outputfolder> -a <num to pick from a> -b <num to pick from b> -c <num to pick from c> --id=<influence data> --noi=<number of iterations> --ape=<ants per edge> --iph=<influence of pheromone> --ecoef=<evaporiation coef> --ssp=<random seed start point>')
                sys.exit()
            elif opt in ("-i", "input_file="):
                input_file = arg
            elif opt in ("-o", "output_path="):
                output_folder = arg
            elif opt in ("-a", "a="):
                how_much_to_choose["a"] = int(arg)
            elif opt in ("-b", "b="):
                how_much_to_choose["b"] = int(arg)
            elif opt in ("-c", "c="):
                how_much_to_choose["c"] = int(arg)
            elif opt in "--id":
                influence_data = int(arg)
            elif opt in "--iph":
                influence_pheromone = float(arg)
            elif opt in "--noi":
                number_of_iterations = int(arg)
            elif opt in "--ape":
                ants_per_edge = int(arg)
            elif opt in "--ecoef":
                evaporation_coef = float(arg)
            elif opt in "--ssp":
                seed_start_point = int(arg)

        if input_file == '':
            raise Exception("Input file was not provided.")
        if output_folder == '':
            raise Exception("Output folder was not provided.")
        if how_much_to_choose['a'] == '':
            raise Exception("Argument 'a' was not provided.")
        if how_much_to_choose['b'] == '':
            raise Exception("Argument 'b' was not provided.")
        if how_much_to_choose['c'] == '':
            raise Exception("Argument 'c' was not provided.")
        if influence_data == '':
            raise Exception("Argument 'id' was not provided.")
        if influence_pheromone == '':
            raise Exception("Argument 'iph' was not provided.")
        if number_of_iterations == '':
            raise Exception("Argument 'noi' was not provided.")
        if ants_per_edge == '':
            raise Exception("Argument 'ape' was not provided.")
        if evaporation_coef == '':
            raise Exception("Argument 'ecoef' was not provided.")
        if seed_start_point == '':
            raise Exception("Argument 'ssp' was not provided.")

    data = converters.JSONToNumpy(input_file)
    group_indices = utilities.get_group_indices(data)
    np.random.seed(seed_start_point)
    array_output_relative_filename = os.path.join(output_folder, "pheromone_array_output.txt")
    tree_output_relative_filename = os.path.join(output_folder, "ants_tree_output.txt")

    data = converters.JSONToNumpy(input_file)
    group_indices = utilities.get_group_indices(data)
    pheromone = np.ones(data.shape)

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

    # for iteration in traversed_edges:
    #     popular_vertices.append()
    #     for key in iteration.keys():

    with open(tree_output_relative_filename, 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            utilities.ptree(-1, ants_tree_dict)

    np.set_printoptions(precision=3, linewidth=300)
    print(pheromone)
    with open(array_output_relative_filename, 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            print(pheromone.round(decimals=3))
    i = 0


if __name__ == "__main__":
    main()
