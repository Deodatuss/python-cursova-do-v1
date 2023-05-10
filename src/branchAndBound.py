import converters
import utilities

import itertools
import numpy as np
import math


def ComputeLowBound(array_data, subarray_slice, fixed_row: list, fixed_column: list, indices: dict, how_much_to_choose: dict, row_bound: str, col_bound: str):
    chosen_triple_tuples = []
    upper_shift = indices[row_bound][0]
    right_shift = indices[col_bound][0]

    confident_row_indices = [
        x for x in fixed_row if x in indices[row_bound]]
    confident_column_indices = [
        x for x in fixed_column if x in indices[col_bound]]

    # first step – compute pairs from confidend row and column indices
    # to find which slots in subarray are 100% going to be used
    all_confident = itertools.product(
        confident_row_indices, confident_column_indices)
    confident_slots = [x for x in all_confident if x[0] != x[1]]

    potential_row_slots = []
    potential_column_slots = []

    for row in [x for x in fixed_row if x in indices[row_bound]]:
        potential_row_slots.append(
            [(row, x) for x in indices[col_bound]]
        )

    for column in [x for x in fixed_column if x in indices[col_bound]]:
        potential_column_slots.append(
            [(x, column) for x in indices[row_bound]]
        )

    how_much_from_rows = np.zeros(array_data.shape[0], dtype=np.int16)
    how_much_from_columns = np.zeros(array_data.shape[1], dtype=np.int16)

    for i in potential_row_slots:
        how_much_from_rows[i[0][0]] = how_much_to_choose[col_bound]

    # number of potential row slots minus already closed confident slots
    # to prevent giving a row more slots than it can have in theory
    for row in potential_row_slots:
        for i in confident_slots:
            if i in row:
                how_much_from_rows[i[0]] -= 1
                row.remove(i)

    for i in potential_column_slots:
        how_much_from_columns[i[0][1]] = how_much_to_choose[row_bound]

    # number of potential column slots minus already closed confident slots
    # to prevent giving a column more slots than it can have in theory
    for column in potential_column_slots:
        for i in confident_slots:
            if i in column:
                how_much_from_columns[i[1]] -= 1
                column.remove(i)

    # first, refill with confident slots
    for i in confident_slots:
        chosen_triple_tuples.append((i[0], i[1], array_data[i[0], i[1]]))

    # then, refill with max elements from potential rows slots minus confident slots
    for row in potential_row_slots:
        potential_row_with_values = [
            (i[0], i[1], array_data[i[0], i[1]]) for i in row
        ]
        potential_row_with_values.sort(key=lambda tup: tup[2], reverse=True)
        for i in zip(potential_row_with_values, range(how_much_from_rows[row[0][0]])):
            chosen_triple_tuples.append(i[0])

    # possibly BUG PRONE (e.g. adding more then some row can handle)
    # do the same for columns
    for column in potential_column_slots:
        potential_column_with_values = [
            (i[0], i[1], array_data[i[0], i[1]]) for i in column
        ]
        potential_column_with_values.sort(key=lambda tup: tup[2], reverse=True)
        for i in zip(potential_column_with_values, range(how_much_from_columns[column[0][1]])):
            chosen_triple_tuples.append(i[0])

    # lastly, if less then needed to choose, find left max slots from subblock
    left_to_choose = (how_much_to_choose[row_bound] *
                      how_much_to_choose[col_bound]) - len(chosen_triple_tuples)

    values_leftover_subtable = np.copy(array_data[subarray_slice])
    for i in chosen_triple_tuples:
        values_leftover_subtable[i[0]-upper_shift,
                                 i[1]-right_shift] = -1

    max_n_indices_in_leftover_subtable = np.argpartition(
        values_leftover_subtable.flat, -left_to_choose
    )[-left_to_choose:]

    data_array_indices = np.unravel_index(
        max_n_indices_in_leftover_subtable, values_leftover_subtable.shape)

    for i in zip(data_array_indices[0], data_array_indices[1], range(left_to_choose)):
        chosen_triple_tuples.append((i[0]+upper_shift,
                                     i[1]+right_shift,
                                     values_leftover_subtable[i[0], i[1]]
                                     ))

    return chosen_triple_tuples


def ChosenElementsToFixedParser(elements_string: str, group_indices: dict):
    """ 
    Parsing string from format like "a1,b2,c3,a5,b1,b3" into fixed rows and columns 
    attached to corresponding elements. 
    """
    fixed_row = []
    fixed_column = []
    elements = elements_string.split(",")

    if elements[0] == "":
        return fixed_row, fixed_column

    for i in elements:
        element_relative_index = int(i[1])-1

        if i[0] == "a":
            fixed_row.append(group_indices[i[0]][element_relative_index])
        elif i[0] == "b":
            fixed_row.append(group_indices[i[0]][element_relative_index])
            fixed_column.append(group_indices[i[0]][element_relative_index])
        elif i[0] == "c":
            fixed_column.append(group_indices[i[0]][element_relative_index])

    return fixed_row, fixed_column


def BoundFromString(array_data, indices, choose_from_groups, element_string: str):
    """
    Uses ComputeLowBound algorithm on ab, ac and bc subarrays to find whole array's low bound
    """
    ab_subarray = np.s_[indices["a"][0]:indices["a"][-1]+1,
                        indices["b"][0]:indices["b"][-1]+1]
    ac_subarray = np.s_[indices["a"][0]:indices["a"][-1]+1,
                        indices["c"][0]:indices["c"][-1]+1]
    bc_subarray = np.s_[indices["b"][0]:indices["b"][-1]+1,
                        indices["c"][0]:indices["c"][-1]+1]

    fixed_row, fixed_column = ChosenElementsToFixedParser(
        element_string, indices)

    ab_list = ComputeLowBound(array_data, ab_subarray, fixed_row,
                              fixed_column, indices, choose_from_groups, "a", "b")
    ac_list = ComputeLowBound(array_data, ac_subarray, fixed_row,
                              fixed_column, indices, choose_from_groups, "a", "c")
    bc_list = ComputeLowBound(array_data, bc_subarray, fixed_row,
                              fixed_column, indices, choose_from_groups, "b", "c")

    sum = 0
    for i in ab_list+ac_list+bc_list:
        sum += i[2]

    return sum


def HighBound(array_data, indices, choose_from_groups):
    return BoundFromString(array_data, indices, choose_from_groups, "")


def BranchAndBound(array_data, indices, choose_from_groups, max_samples_for_branch=1, start_level=0):
    def BndBCheckAndUpdate(key, fields, dict_to_update) -> None:
        """
        algorithm that forces choosing consecutive elements from different groups,
        i.e. after first group 'a' second level of a tree should be 'ab' or 'ac' but not 'aa'
        """
        for element in all_entries:
            # discard already used elements
            if element in key_elems:
                continue

            already_used_in_group = fields["in_groups_left"]
            # discard using empty groups
            if already_used_in_group[element[0]] == 0:
                continue

            add_new = key+","+element

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

            new_groups_left = fields["in_groups_left"].copy()
            new_groups_left[element[0]] -= 1

            # don't sort – it's against tree algorithm!
            # sorted_new = ','.join(sorted(add_new.split(",")))

            dict_to_update.update({add_new: {
                "value": np.round(BoundFromString(
                    array_data, indices, choose_from_groups, add_new), 3),
                "in_groups_left":  new_groups_left
            }})

    all_entries = []
    traversed_tree = []
    for key, value in indices.items():
        for i in zip(value, range(len(value))):
            all_entries.append(key+str(i[1]+1))

    # first tree level: finding low bounds for every element
    level_results = {}
    for element in all_entries:
        already_used_in_group = choose_from_groups.copy()
        already_used_in_group[element[0]] -= 1
        level_results.update({element: {
            "value": BoundFromString(
                array_data, indices, choose_from_groups, element),
            "in_groups_left": already_used_in_group
        }
        })

    when_stopping_full_search_counter = 0

    # range is a depth of the tree, starting from second tree level
    for i in range(sum(choose_from_groups.values())-1):
        max_unique_keys = []
        selected_keys_to_traverse = {}

        # setting to continue level with all possible values i.e. full search
        if when_stopping_full_search_counter < start_level:
            max_unique_keys = [x for x in level_results.keys()]
            when_stopping_full_search_counter += 1
            for key in max_unique_keys:
                selected_keys_to_traverse.update({key: level_results[key]})

        # setting to continue level with top max_samples_for_branch highest value(s) + all dublicates
        else:
            max_unique_keys = sorted(
                level_results, key=lambda x: level_results[x]["value"]
            )
            max_values = [level_results[x]["value"] for x in max_unique_keys]

            # top max_samples_for_branch highest value(s)
            max_values = np.unique(np.array(max_values)
                                   )[-max_samples_for_branch:]

            # find and add any max value duplicates (alternative paths)
            for key in max_unique_keys:
                if level_results[key]["value"] in max_values:
                    selected_keys_to_traverse.update(
                        {key: level_results[key]})

        level_results = {}
        for key, fields in selected_keys_to_traverse.items():
            recent_key_group = key[-(2 % len(key))]
            second_recent_key_group = key[-(5 % len(key))]
            key_elems = key.split(",")

            BndBCheckAndUpdate(key, fields, level_results)
        traversed_tree.append(selected_keys_to_traverse)

    traversed_tree.append(level_results)

    return traversed_tree


def DataDictToTreedictConverter(dictionary):
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


def main():
    """
    This function is used as a test and presentation for file functions
    """
    folder = "data/"
    input_relative_filename = folder + "input.json"
    output_relative_filename = folder + "output.json"

    test_indices = {
        'a': [0, 1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10], 'c': [11, 12, 13, 14]
    }

    how_much_to_choose = {
        'a': 2, 'b': 3, 'c': 1
    }
    data = converters.JSONToNumpy(input_relative_filename)
    groups_info = utilities.GetGroupsSize(data)

    # indices of elements from each group
    # e.g. [0, 1], [2, 3], [4, 5] if array of 6 elements was divided in 3 groups by 2
    a = [groups_info["a"]["size"], groups_info["b"]
         ["size"], groups_info["c"]["size"]]

    group_A = [x for x in range(a[0])]
    group_B = [x for x in range(a[0], a[0]+a[1])]
    group_C = [x for x in range(a[0]+a[1], a[0]+a[1]+a[2])]

    group_indices = {
        "a": group_A,
        "b": group_B,
        "c": group_C
    }

    HB = HighBound(data, group_indices, how_much_to_choose)

    tr_tree = BranchAndBound(data, group_indices, how_much_to_choose)

    tree_dict = {
        -1: ["A1; 10,2", "A2; 9,8", "A3; 10,0"],
        "A1; 10,2": ["A1-B1; 8,2", "A1-B4; 8,2"],
        "A3; 10,0": ["A3-B1; 8,4", "A3-B3; 8,7"],
        "A2; 9,8": ["X"],
        "A1-B4; 8,2": ["A1-B4-C1; 7,2", "A1-B4-C2; 7,5", "A1-B4-C3; 7,3"]
    }

    final_tree, max_dict = DataDictToTreedictConverter(tr_tree)
    utilities.ptree(-1, final_tree, indent_width=9)


if __name__ == "__main__":
    main()
