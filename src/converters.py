import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def NumpyToJSON(numpy_array, filename="input.json"):
    numpy_data = {"array": numpy_array}
    with open(filename, "w") as f:
        json.dump(numpy_data, f, cls=NumpyArrayEncoder, indent=4)
    return filename


def JSONToArray(filename, array_key="array"):
    data = {}

    with open(filename, "r") as f:
        data = json.load(f)
    return data[array_key]


def JSONToNumpy(filename, array_key="array"):
    return np.array(JSONToArray(filename, array_key))


def CSVToNumpy(filename):
    return np.genfromtxt(filename, delimiter=',')


def DictToJSONFile(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
