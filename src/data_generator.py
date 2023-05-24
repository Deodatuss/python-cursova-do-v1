import random

import numpy as np


def generate_compatibility_matrix(a_size: int, b_size: int, c_size: int):
    matrix_size = a_size + b_size + c_size
    compatibility_matrix = [[0 for x in range(matrix_size)] for y in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(i, matrix_size):
            value = round(random.normalvariate(0.5, 0.25), 2)
            if value < 0:
                compatibility_matrix[i][j] = 0
            else:
                compatibility_matrix[i][j] = value

    for i in range(matrix_size):
        for j in range(i, matrix_size):
            compatibility_matrix[j][i] = compatibility_matrix[i][j]

    for i in range(a_size):
        for j in range(a_size):
            compatibility_matrix[i][j] = 1

    for i in range(a_size, a_size + b_size):
        for j in range(a_size, a_size + b_size):
            compatibility_matrix[i][j] = 1

    for i in range(a_size + b_size, a_size + b_size + c_size):
        for j in range(a_size + b_size, a_size + b_size + c_size):
            compatibility_matrix[i][j] = 1

    return np.array(compatibility_matrix)
