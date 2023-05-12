import utilities
import numpy as np
import itertools


def UniqueThreeMax(array_data, indices, choose_from_groups):
    pass
    # find which array intersects at 3-2, 3-1 or 2-1
    # (e.g. for 3-2-1 choosing the AB will be 3-2, AC 3-1, and BC 2-1)
    choosed_values = list(choose_from_groups.values())
    choosed_keys = list(choose_from_groups.keys())
    subarray32 = ''.join(sorted(
        choosed_keys[choosed_values.index(3)] +
        choosed_keys[choosed_values.index(2)]))
    subarray31 = ''.join(sorted(
        choosed_keys[choosed_values.index(3)] +
        choosed_keys[choosed_values.index(1)]))
    subarray21 = ''.join(sorted(
        choosed_keys[choosed_values.index(2)] +
        choosed_keys[choosed_values.index(1)]))

    intersected_subarrays = [subarray32, subarray31, subarray21]

    ab_subarray = {'name': 'ab',
                   'slice': np.s_[indices["a"][0]:indices["a"][-1]+1,
                                  indices["b"][0]:indices["b"][-1]+1]}
    ac_subarray = {'name': 'ac',
                   'slice': np.s_[indices["a"][0]:indices["a"][-1]+1,
                                  indices["c"][0]:indices["c"][-1]+1]}
    bc_subarray = {'name': 'bc',
                   'slice': np.s_[indices["b"][0]:indices["b"][-1]+1,
                                  indices["c"][0]:indices["c"][-1]+1]}

    subarray_slices = [ab_subarray, ac_subarray, bc_subarray]

    first_max_element_position = []
    second_max_element_position = []
    third_max_element_position = []

    # for subarrays:
    #   from subarray intersected at 3-2 get 2 max elements from non-matching rows and columns
    for slice in subarray_slices:
        if slice['name'] == subarray32:
            # get indices shift for subarray, so max elements would have indices
            # corresponding to the full array
            upper_shift = indices[subarray32[0]][0]
            right_shift = indices[subarray32[1]][0]

            # sort subarray and get indices
            subarray = array_data[slice['slice']]
            sorted_flat_indices = np.flip(np.argsort(subarray.ravel()))
            sorted_2d_indices = np.unravel_index(
                sorted_flat_indices, subarray.shape)

            # first, get max element position and value
            first_max_element_position = [
                sorted_2d_indices[0][0],
                sorted_2d_indices[1][0]
            ]
            array = array_data[tuple(first_max_element_position)]

            # then, get second largest element with rows and columns different from first's
            stacked_2d_indices = np.dstack(sorted_2d_indices)
            for position in stacked_2d_indices[0]:
                this_x = position[1]
                max_x = first_max_element_position[1]
                this_y = position[0]
                max_y = first_max_element_position[0]
                if this_x != max_x and this_y != max_y:
                    second_max_element_position = position
                    break

            # update fist and second max elements with absolute indices instead of
            # current subarray's relative indices
            first_max_element_position = [
                first_max_element_position[0]+upper_shift,
                first_max_element_position[1]+right_shift
            ]
            second_max_element_position = [
                second_max_element_position[0]+upper_shift,
                second_max_element_position[1]+right_shift
            ]

    #   from subarray intersected at 3-1 get 1 max element that doesn't match with previous two
    for slice in subarray_slices:
        if slice['name'] == subarray31:
            # get indices shift for new subarray
            upper_shift = indices[subarray31[0]][0]
            right_shift = indices[subarray31[1]][0]

            # sort new subarrray
            subarray = array_data[slice['slice']]
            sorted_flat_indices = np.flip(np.argsort(subarray.ravel()))
            sorted_2d_indices = np.unravel_index(
                sorted_flat_indices, subarray.shape)
            stacked_2d_indices = np.dstack(sorted_2d_indices)

            # element that doesn't match with previous two's indices
            for position in stacked_2d_indices[0]:
                this_x = position[1]
                first_max_x = first_max_element_position[1]
                second_max_x = second_max_element_position[1]

                this_y = position[0]
                first_max_y = first_max_element_position[0]
                second_max_y = second_max_element_position[0]
                if this_x != first_max_x != second_max_x and \
                        this_y != first_max_y != second_max_y:
                    third_max_element_position = position
                    break

            # update third max element with absolute indices
            third_max_element_position = [
                third_max_element_position[0]+upper_shift,
                third_max_element_position[1]+right_shift
            ]

    return [tuple(first_max_element_position),
            tuple(second_max_element_position),
            tuple(third_max_element_position)]


def GreedyValue(array_data, indices, choose_from_groups):
    three_unique_max = UniqueThreeMax(array_data, indices, choose_from_groups)

    all_indices = [item for sublist in three_unique_max for item in sublist]

    all_indices = sorted(all_indices)

    all_elements = list(itertools.product(all_indices, repeat=2))

    full_greedy_value = 0

    for el in all_elements:
        i = array_data[el]
        full_greedy_value += i

    partial_greedy_value = utilities.FullValueToPartial(
        full_greedy_value, choose_from_groups)

    return partial_greedy_value
