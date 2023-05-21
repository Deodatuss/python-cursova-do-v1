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
    input_folder = "data/demo/"
    output_folder = "data/demo/ants/"

    input_relative_filename = input_folder + "input.json"
    array_output_relative_filename = output_folder + "pheromone_array_output.txt"
    tree_output_relative_filename = output_folder + "ants_tree_output.txt"

    # number of elements from each of three groups
    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }
    number_of_iterations = 15
    ants_per_edge = 2
    influence_data = 1
    influence_pheromone = 0.8
    evaporation_coef = 0.1
    np.random.seed(0)

    data = converters.JSONToNumpy(input_relative_filename)
    group_indices = utilities.GetGroupIndices(data)
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
        _, iter_max_ant = bnb.DataDictToTreedictConverter(
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

    ants_tree_dict, _ = bnb.DataDictToTreedictConverter(
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
