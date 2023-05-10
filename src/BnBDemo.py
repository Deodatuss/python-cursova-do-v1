import branchAndBound as bnb
import converters
import utilities
from contextlib import redirect_stdout


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
        'a': 1, 'b': 3, 'c': 2
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

    HB = bnb.HighBound(data, group_indices, how_much_to_choose)

    tr_tree = bnb.BranchAndBound(
        data, group_indices, how_much_to_choose, max_samples_for_branch=1, start_level=0)

    tree_dict = {
        -1: ["A1; 10,2", "A2; 9,8", "A3; 10,0"],
        "A1; 10,2": ["A1-B1; 8,2", "A1-B4; 8,2"],
        "A3; 10,0": ["A3-B1; 8,4", "A3-B3; 8,7"],
        "A2; 9,8": ["X"],
        "A1-B4; 8,2": ["A1-B4-C1; 7,2", "A1-B4-C2; 7,5", "A1-B4-C3; 7,3"]
    }

    final_tree, max_dict = bnb.DataDictToTreedictConverter(tr_tree)

    with open('tree_output.txt', 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            utilities.ptree(-1, final_tree, indent_width=9)


if __name__ == "__main__":
    main()
