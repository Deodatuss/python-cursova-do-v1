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

# numpyArrayOne = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])

# arrayTwo = CSVToNumpy("src/importData.csv")

# print(arrayTwo)

# _ = NumpyToJSON(arrayTwo)

# arrayThree = JSONToArray("results.json")

# print(np.array(arrayThree))

# # Serialization
# numpyData = {"array": numpyArrayOne}
# # use dump() to write array into file
# encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
# print("Printing JSON serialized NumPy array")
# print(encodedNumpyData)

# # Deserialization
# print("Decode JSON serialized NumPy array")
# decodedArrays = json.loads(encodedNumpyData)

# finalNumpyArray = np.asarray(decodedArrays["array"])
# print("NumPy Array")
# print(finalNumpyArray)
