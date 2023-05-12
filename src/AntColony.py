import numpy as np
import converters
import utilities
import GreedyAlgorithm


def AllPossibleStates(element_array, pheromone_array, element_influence, pheromone_influence):
    all_states_sum = 0
    for pair in zip(element_array, pheromone_array):
        all_states_sum += (pair[0]**element_influence) * \
            (pair[1]**pheromone_influence)

    return all_states_sum


def ProbabilityOfMovingToState(element_value, pheromone_value, element_influence, pheromone_influence, all_states):
    p = ((element_value**element_influence) *
         (pheromone_value**pheromone_influence))/all_states

    return p


def AntIteration(array_data, indices, choose_from_groups, array_pheromone, ants_count):
    all_entries = []
    for key, value in indices.items():
        for i in zip(value, range(len(value))):
            all_entries.append(key+str(i[1]+1))

    ants = [[]*ants_count]

    # for i in range(sum(choose_from_groups.values())):


def main():
    """
    This function is used as a test and presentation for Ant Colony Algorithm functions and workflow
    """
    input_folder = "data/demo/"
    output_folder = "data/demo/"

    input_relative_filename = input_folder + "input.json"
    dict_output_relative_filename = output_folder + "dict_output.json"
    tree_output_relative_filename = output_folder + "tree_output.txt"

    # number of elements from each of three groups
    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }

    data = converters.JSONToNumpy(input_relative_filename)
    group_indices = utilities.GetGroupIndices(data)
    pheromone = np.ones(data.shape)

    GreedyAlgorithm.GreedyValue(data, group_indices, how_much_to_choose)

    AntIteration(data, group_indices, how_much_to_choose, pheromone, 2)


if __name__ == "__main__":
    main()
