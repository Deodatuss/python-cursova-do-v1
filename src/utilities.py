def get_groups_size(numpy_array):
    result = {
        "a":
        {
            "size": 0
        },
        "b":
        {
            "size": 0
        },
        "c":
        {
            "size": 0
        }
    }
    counter = 0

    while (numpy_array[0][counter] == 1):
        result["a"]["size"] += 1
        counter += 1

    while (numpy_array[result["a"]["size"]][counter] == 1):
        result["b"]["size"] += 1
        counter += 1

    result["c"]["size"] = numpy_array.shape[0] - \
        (result["a"]["size"]+result["b"]["size"])

    return result


def get_group_indices(data):
    group_size = get_groups_size(data)
    """
    return group_indices in such format and keys:
    group_indices = {
        'a': [0, 1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10], 'c': [11, 12, 13, 14]
    }
    """
    # indices of elements from each group
    # e.g. [0, 1], [2, 3], [4, 5] if array of 6 elements was divided in 3 groups by 2
    a = [group_size["a"]["size"], group_size["b"]
         ["size"], group_size["c"]["size"]]

    group_a = [x for x in range(a[0])]
    group_b = [x for x in range(a[0], a[0]+a[1])]
    group_c = [x for x in range(a[0]+a[1], a[0]+a[1]+a[2])]

    group_indices = {
        "a": group_a,
        "b": group_b,
        "c": group_c
    }

    return group_indices


def partial_value_to_full(partial_value: float, choose_from_groups) -> float:
    full_value = (partial_value*2) + (
        choose_from_groups['a']**2 +
        choose_from_groups['b']**2 +
        choose_from_groups['c']**2)

    return full_value


def full_value_to_partial(full_value: float, choose_from_groups) -> float:
    partial_value = ((full_value -
                      choose_from_groups['a']**2 -
                      choose_from_groups['b']**2 -
                      choose_from_groups['c']**2)/2)

    return partial_value


def ptree(start, tree, indent_width=4):
    """
    https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command
    """
    def _ptree(start, parent, tree, grandpa=None, indent=""):
        if parent != start:
            if grandpa is None:  # Ask grandpa kids!
                print(parent, end="")
            else:
                print(parent)
        if parent not in tree:
            return
        for child in tree[parent][:-1]:
            print(indent + "├" + "─" * indent_width, end="")
            _ptree(start, child, tree, parent,
                   indent + "│" + " " * indent_width)
        child = tree[parent][-1]
        print(indent + "└" + "─" * indent_width, end="")
        _ptree(start, child, tree, parent, indent +
               " " * (indent_width+2))  # 4 -> 5

    parent = start
    print(start)
    _ptree(start, parent, tree)
