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


def IndexFromString(position, indices):
    group = position[0]
    local_position = int(position[1])
    for key in indices.keys():
        if group == key:
            a = indices[key]


def MoveValue(array_data, indices, new_ant):
    path = list(new_ant.keys())[0]
    new_move = path[-(2 % len(path))]
    current_pos = path[-(2 % len(path))]

    y = IndexFromString(new_move, indices)
    x = IndexFromString(current_pos, indices)


def PossibleNewPathsForAnt(ant, all_entries):
    path = list(ant.keys())[0]
    recent_key_group = path[-(2 % len(path))]
    second_recent_key_group = path[-(5 % len(path))]
    key_elems = path.split(",")

    possible_paths = []
    for element in all_entries:
        # discard already used elements
        if element in key_elems:
            continue

        already_used_in_group = ant[path]["in_groups_left"]
        # discard using empty groups
        if already_used_in_group[element[0]] == 0:
            continue

        possible_new_ant = path+","+element

        big_loop_violation = False
        small_loop_violation = False
        if element[0] == second_recent_key_group:
            big_loop_violation = True

        if element[0] == recent_key_group:
            small_loop_violation = True

        big_loop_can_be_made = False

        if big_loop_violation:
            for group in already_used_in_group:
                if group != recent_key_group \
                        and group != second_recent_key_group \
                        and already_used_in_group[group] > 0:
                    big_loop_can_be_made = True

        small_loop_can_be_made = False
        if small_loop_violation:
            for group in already_used_in_group:
                if group != recent_key_group \
                        and already_used_in_group[group] > 0:
                    small_loop_can_be_made = True

        if (big_loop_violation and big_loop_can_be_made):
            continue

        if (small_loop_violation and small_loop_can_be_made):
            continue

        possible_paths.append(element)

    return possible_paths


def ChooseNextPath(array_data, indices, array_pheromone, all_entries, ant):
    possible_moves = PossibleNewPathsForAnt(ant, all_entries)


def AntIteration(array_data, indices, choose_from_groups, array_pheromone, ants_per_element=1):
    all_entries = []
    for key, value in indices.items():
        for i in range(len(value)):
            all_entries.append(key+str(i+1))

    ants = []
    # start by placing ants on elements
    for element in all_entries:
        already_used_in_group = choose_from_groups.copy()
        already_used_in_group[element[0]] -= 1

        for ant in range(ants_per_element):
            ants.append({element: {
                "path_value": 0,
                "in_groups_left": already_used_in_group
            }})

    # second layer
    current_ant = ants[0]
    ChooseNextPath(array_data, indices, array_pheromone, all_entries, ants[0])
    PossibleNewPathsForAnt(ants[0], all_entries)
    # for i in range(sum(choose_from_groups.values())-1):


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
