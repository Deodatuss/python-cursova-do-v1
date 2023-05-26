import converters
import utilities
import itertools
import numpy as np

folder = "data/"
input_relative_filename = folder + "input.json"
output_relative_filename = folder + "bruteforce_output.json"
how_much_to_choose = [2, 3, 1]

# generates all possible permutations for given how_much_to_choose
#  i.e. even if given [2, 3, 1],  it will anyway also generate and use [3, 2, 1], [2, 1, 3] etc

elements_to_choose_from_each_group = list(
    itertools.permutations(how_much_to_choose))
elements_to_choose_from_each_group = [how_much_to_choose]

data = converters.JSONToNumpy(input_relative_filename)

groups_info = utilities.get_groups_size(data)

a = [groups_info["a"]["size"], groups_info["b"]
     ["size"], groups_info["c"]["size"]]


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

    summed += all_combinations


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
corr_effectiveness = np.zeros((summed, 1), np.float32)
data_indeces = [x for x in range(a[0]+a[1]+a[2])]

counter = 0
for row in pos_combinations:
    sum = 0
    mask = np.isin(data_indeces, row)

    for index in row:
        sum += np.sum(data[index][mask])

    sum = (sum - 14)/2

    corr_effectiveness[counter] = sum
    counter += 1

unique, counts = np.unique(corr_effectiveness, return_counts=True)

unique_counts = {}
for item in zip(unique, counts):
    unique_counts.update({str(item[0]): int(item[1])})

print(unique_counts)

converters.DictToJSONFile(unique_counts, output_relative_filename)
