def GetGroupsSize(numpy_array):
    result = {
        "group_A":
        {
            "size": 0
        },
        "group_B":
        {
            "size": 0
        },
        "group_C":
        {
            "size": 0
        }
    }
    counter = 0

    while (numpy_array[0][counter] == 1):
        result["group_A"]["size"] += 1
        counter += 1

    while (numpy_array[result["group_A"]["size"]][counter] == 1):
        result["group_B"]["size"] += 1
        counter += 1

    result["group_C"]["size"] = numpy_array.shape[0] - \
        (result["group_A"]["size"]+result["group_B"]["size"])

    return result
