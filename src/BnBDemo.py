import getopt
import os
import sys

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
    input_file = ''
    output_path = ''
    # number of elements from each of three groups
    how_much_to_choose = {
        'a': '', 'b': '', 'c': ''
    }
    max_samples_for_branch = ''
    start_level = ''

    opts, args = getopt.getopt(sys.argv[1:], "hi:d:o:a:b:c:",
                               ["input_file=",
                                "output_path=",
                                "a=", "b=", "c=",
                                "max_samples_for_branch=",
                                "msb=",
                                "sl=",
                                "start_level="])

    is_default_values = next((arg for opt, arg in opts if opt == "-d" and arg == 'true'), None)

    if(is_default_values is not None):
        root_folder = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(root_folder, '..', 'data', 'demo', "importData.csv")
        output_path = os.path.join(root_folder, '..', 'data', 'demo')
        # number of elements from each of three groups
        how_much_to_choose = {
            'a': 1, 'b': 3, 'c': 2
        }
        max_samples_for_branch = 1
        start_level = 0
    else:
        for opt, arg in opts:
            if opt == '-h':
                print(
                    'python BnBDemo.py -i <inputfile.csv> -o <outputpath> -a <num to pick from a> -b <num to pick from b> -c <num to pick from c> --msb=<max samples for branch> -sl=<start level>')
                sys.exit()
            elif opt in ("-i", "input_file="):
                input_file = arg
            elif opt in ("-o", "output_path="):
                output_path = arg
            elif opt in ("-a", "a="):
                how_much_to_choose["a"] = int(arg)
            elif opt in ("-b", "b="):
                how_much_to_choose["b"] = int(arg)
            elif opt in ("-c", "c="):
                how_much_to_choose["c"] = int(arg)
            elif opt in ("--msb", "max_samples_for_branch="):
                max_samples_for_branch = int(arg)
            elif opt in ("--sl", "start_level="):
                start_level = int(arg)

        if(input_file == ''):
            raise Exception("Input file was not provided.")
        if (output_path == ''):
            raise Exception("Output file was not provided.")
        if (how_much_to_choose['a'] == ''):
            raise Exception("Argument 'a' was not provided.")
        if (how_much_to_choose['b'] == ''):
            raise Exception("Argument 'b' was not provided.")
        if (how_much_to_choose['c'] == ''):
            raise Exception("Argument 'c' was not provided.")
        if (max_samples_for_branch == ''):
            raise Exception("Argument 'msb' was not provided.")
        if (start_level == ''):
            raise Exception("Argument 'sl' was not provided.")

    dict_output_relative_filename = os.path.join(output_path, "dict_output.json")
    tree_output_relative_filename = os.path.join(output_path, "tree_output.txt")

    data = converters.CSVToNumpy(input_file)
    group_indices = utilities.GetGroupIndices(data)

    high_bound = bnb.HighBound(data, group_indices, how_much_to_choose)

    tr_tree = bnb.BranchAndBound(
        data,
        group_indices,
        how_much_to_choose,
        max_samples_for_branch,
        start_level)

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

    print('Done. Find output files in ' + output_path)

if __name__ == "__main__":
    main()
