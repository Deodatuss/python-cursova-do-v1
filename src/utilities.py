def GetGroupsSize(numpy_array):
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
