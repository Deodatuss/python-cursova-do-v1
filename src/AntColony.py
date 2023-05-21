import numpy as np
from branchAndBound import bound_from_string


def ProbabilityOfAllPaths(array_data, array_pheromone, influence_data, influence_pheromone, paths: dict):
    data_times_pheromone = {}
    sum_probability = 0
    for path, position in paths.items():
        probability = MoveSimpleProbability(
            array_data[position], array_pheromone[position], influence_data, influence_pheromone)
        data_times_pheromone.update({path: probability})
        sum_probability += probability

    edge_selection_probability = []
    paths = []
    for path, probability in data_times_pheromone.items():
        edge_selection_probability.append(probability/sum_probability)
        paths.append(path)

    return edge_selection_probability, paths


def MoveSimpleProbability(value_data, value_pheromone, influence_data, influence_pheromone):
    return (value_data**influence_data) * \
        (value_pheromone**influence_pheromone)


def IndexFromString(position, indices):
    group = position[0]
    local_position = int(position[1])
    for key in indices.keys():
        if group == key:
            return indices[key][local_position-1]


def MoveIndex(indices, new_path):
    new_move = new_path[-(2 % len(new_path)):]
    current_pos = new_path[-(5 % len(new_path)):-(5 % len(new_path))+2]

    y = IndexFromString(new_move, indices)
    x = IndexFromString(current_pos, indices)
    return (y, x)


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

        possible_new_ant = path+","+element
        possible_paths.append(possible_new_ant)

    return possible_paths


def PossibleNextMoveIndex(array_data, indices, array_pheromone, all_entries, ant):
    possible_paths = PossibleNewPathsForAnt(ant, all_entries)
    path_indices_to_new_element = {}
    for path in possible_paths:
        index = MoveIndex(indices, path)
        path_indices_to_new_element.update({path: index})
    return path_indices_to_new_element


def AntIteration(array_data, indices, choose_from_groups, array_pheromone,
                 ants_per_element, influence_data, influence_pheromone):
    return_array_of_dict = []

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
                "value": 0,
                "in_groups_left": already_used_in_group
            }})

    temp_dict = {}
    for ant in ants:
        temp_dict.update(ant)
    return_array_of_dict.append(temp_dict)

    for layer in range(sum(choose_from_groups.values())-1):
        ant_iterator = 0
        for ant in ants:
            paths_and_indices = PossibleNextMoveIndex(array_data, indices, array_pheromone,
                                                      all_entries, ant)
            probabilities, paths_only = ProbabilityOfAllPaths(
                array_data, array_pheromone, influence_data, influence_pheromone, paths_and_indices)

            # probabilities = [0.92, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

            random_number = np.random.rand(1)
            chosen_path_index = 0
            sum_of_probabilites = 0
            for prob in zip(probabilities, range(len(probabilities))):
                sum_of_probabilites += prob[0]
                if sum_of_probabilites > random_number:
                    chosen_path_index = prob[1]
                    break

            new_ant_path = paths_only[chosen_path_index]
            added_value = array_data[MoveIndex(indices, new_ant_path)]

            for key, values in ant.items():
                new_groups_left = values["in_groups_left"].copy()
                new_groups_left[new_ant_path[-2]] -= 1
                new_ant = {new_ant_path: {
                    "value": values['value']+added_value,
                    "in_groups_left": new_groups_left
                }}
                ants[ant_iterator] = new_ant

            ant_iterator += 1
        temp_dict = {}
        for ant in ants:
            temp_dict.update(ant)
        return_array_of_dict.append(temp_dict)

    return return_array_of_dict


def DataDictToPreferablePathsTreeDict(dictionary):
    """
    Reformats BranchAndBound dictionary into other dictionary used to build a tree
    """
    tree_compliant_arr = []
    tree_compliant_dict = {-1: "X"}

    for level in dictionary:
        for key, value in level.items():
            tree_compliant_arr.append([key, value["value"]])
            tree_compliant_dict.update({key+": "+str(value["value"]): "X"})

    # populate root of the tree
    add_array = []
    for element, value in dictionary[0].items():
        add_array.append(element+": "+str(value["value"]))
    tree_compliant_dict.update({-1: add_array})

    # populate branches
    for i in range(0, len(dictionary)-1):
        for row, value in dictionary[i].items():
            add_array = []
            for next_row, next_value in dictionary[i+1].items():
                if row == next_row[:len(row)]:
                    add_array.append(next_row+": "+str(next_value["value"]))
            if len(add_array) > 0:
                tree_compliant_dict.update(
                    {row+": "+str(value["value"]): add_array})

    max_value = -1
    max_dict = {}
    max_dict_key = ""
    for i, fields in dictionary[-1].items():
        if fields["value"] > max_value:
            max_dict = {i: fields}
            max_value = fields["value"]
            max_dict_key = i+": "+str(fields["value"])

    tree_compliant_dict[max_dict_key] = "O"
    return tree_compliant_dict, max_dict


def TraversedEdgesFromString(indices, element_string):
    vertices = element_string.split(",")
    TraversedEdgesFromString = []
    for index_tuple in zip(range(0, len(vertices)-1), range(1, len(vertices))):
        x_position = IndexFromString(vertices[index_tuple[0]], indices)
        y_position = IndexFromString(vertices[index_tuple[1]], indices)

        TraversedEdgesFromString.append((x_position, y_position))
    return TraversedEdgesFromString


def UpdatePheromoneArray(array_data, indices, choose_from_groups, array_pheromone, ant_paths, evaporation_coef):
    full_paths = ant_paths[-1]
    values = []
    max_value = 0

    # negative feedback on pheromones
    array_pheromone *= (1-evaporation_coef)

    # get values which shows how lucrative every path is
    for key in full_paths.keys():
        ant_value = bound_from_string(
            array_data, indices, choose_from_groups, key)
        values.append(ant_value)
        if ant_value > max_value:
            max_value = ant_value

    # positive feedback on pheromones
    for key_and_value in zip(full_paths.keys(), values):
        edges = TraversedEdgesFromString(indices, key_and_value[0])
        for edge in edges:
            array_pheromone[edge] += key_and_value[1]/max_value
            # change mirrored element too
            array_pheromone[edge[1], edge[0]] += key_and_value[1]/max_value


def AddEdgeTraversesToStrings(ants_paths, traversed_edges, indices):
    elements_plus_edges = [ants_paths[0]]
    for level in ants_paths[1:]:
        elements_plus_edges.append({})
        for key, values in level.items():
            elements = key.split(",")
            el_indices = [IndexFromString(x, indices) for x in elements]
            new_string = ""
            for i in range(len(elements)-1):
                new_string += elements[i]
                new_string += '-'

                ind_tuple = (el_indices[i], el_indices[i+1])
                rever_tuple = (el_indices[i+1], el_indices[i])

                pos_value = traversed_edges.get(ind_tuple)
                pos_rever_value = traversed_edges.get(rever_tuple)

                if pos_value:
                    new_string += str(pos_value)
                else:
                    new_string += str(pos_rever_value)
                new_string += '-'

            new_string += elements[-1]
            elements_plus_edges[-1].update({new_string: values})
    return elements_plus_edges
