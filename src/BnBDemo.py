import os

import branchAndBound as bnb
import converters
import utilities
from contextlib import redirect_stdout
import json


def main():
    """
    This function is used as a test and presentation for BnB functions and workflow
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    input_relative_filename = os.path.join(root_folder, '..', 'data', 'demo', "importData.csv")
    dict_output_relative_filename = os.path.join(root_folder, '..', 'data', 'demo', "dict_output.json")
    tree_output_relative_filename = os.path.join(root_folder, '..', 'data', 'demo', "tree_output.txt")

    # number of elements from each of three groups
    how_much_to_choose = {
        'a': 1, 'b': 3, 'c': 2
    }

    data = converters.CSVToNumpy(input_relative_filename)
    group_indices = utilities.GetGroupIndices(data)

    high_bound = bnb.HighBound(data, group_indices, how_much_to_choose)

    tr_tree = bnb.BranchAndBound(
        data, group_indices, how_much_to_choose, max_samples_for_branch=1, start_level=0)

    dict_for_json = {}

    for level in tr_tree:
        dict_for_json.update(level)

    # # save tr_tree dict to json file
    with open(dict_output_relative_filename, 'w') as fp:
        json.dump(dict_for_json, fp)

    final_tree, max_dict = bnb.DataDictToTreedictConverter(tr_tree)

    # final tree has a structure similar to this:
    # test_tree_dict = {
    #     -1: ["A1; 10,2", "A2; 9,8", "A3; 10,0"],
    #     "A1; 10,2": ["A1-B1; 8,2", "A1-B4; 8,2"],
    #     "A3; 10,0": ["A3-B1; 8,4", "A3-B3; 8,7"],
    #     "A2; 9,8": ["X"],
    #     "A1-B4; 8,2": ["A1-B4-C1; 7,2", "A1-B4-C2; 7,5", "A1-B4-C3; 7,3"]
    # }

    with open(tree_output_relative_filename, 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            utilities.ptree(-1, final_tree, indent_width=9)


if __name__ == "__main__":
    main()
