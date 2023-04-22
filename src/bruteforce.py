import converters
import utilities

import itertools
import numpy as np

folder = "src/"
input_relative_filename = folder + "input.json"
output_relative_filename = folder + "output.json"
how_much_to_choose = [1, 2, 3]
elements_to_choose_from_each_group = list(
    itertools.permutations(how_much_to_choose))

data = converters.JSONToNumpy(input_relative_filename)

groups_info = utilities.GetGroupsSize(data)

a = [groups_info["group_A"]["size"], groups_info["group_B"]
     ["size"], groups_info["group_C"]["size"]]

# indices of elements from each group
# e.g. [0, 1], [2, 3], [4, 5] if array of 6 elements was divided in 3 groups by 2
group_A = [x for x in range(a[0])]
group_B = [x for x in range(a[0], a[0]+a[1])]
group_C = [x for x in range(a[0]+a[1], a[0]+a[1]+a[2])]


# count number of possible combinations
summed = 0
for i in elements_to_choose_from_each_group:
    all_combinations = len(list(itertools.combinations(group_A, i[0])))
    all_combinations *= len(list(itertools.combinations(group_B, i[1])))
    all_combinations *= len(list(itertools.combinations(group_C, i[2])))

    # print("for ", i, ": ", all_combinations)
    summed += all_combinations
# print(summed)


# gather all possible combinations
pos_combinations = np.zeros((summed, sum(how_much_to_choose)), np.int8)

row = 0
for i in elements_to_choose_from_each_group:
    cm_a = list(itertools.combinations(group_A, i[0]))
    cm_b = list(itertools.combinations(group_B, i[1]))
    cm_c = list(itertools.combinations(group_C, i[2]))
    index_a = [x for x in range(i[0])]
    index_b = [x for x in range(i[0], i[0]+i[1])]
    index_c = [x for x in range(i[0]+i[1], i[0]+i[1]+i[2])]

    for j1 in cm_a:
        for j2 in cm_b:
            for j3 in cm_c:
                np.put(pos_combinations[row], index_a, j1)
                np.put(pos_combinations[row], index_b, j2)
                np.put(pos_combinations[row], index_c, j3)
                row += 1


# count corresponded row-by-row effectivenes value for all possible combinations
corr_effectiveness = np.zeros((summed, 1), np.float16)
data_indeces = [x for x in range(a[0]+a[1]+a[2])]

counter = 0
for row in pos_combinations:
    sum = 0
    mask = np.isin(data_indeces, row)

    for index in row:
        sum += np.sum(data[index][mask])

    corr_effectiveness[counter] = sum
    counter += 1

for i in corr_effectiveness:
    print("{:.4f}".format(i[0]))

unique, counts = np.unique(corr_effectiveness, return_counts=True)

print(dict(zip(unique, counts)))
