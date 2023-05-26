import random
import string

import numpy as np


def generate_compatibility_matrix(
        a_size: int,
        b_size: int,
        c_size: int,
        generate_method: string = 'normalvariate',
        mean: float = 0.5,
        dispersion: float = 0.5):
    def generate_value(generate_method: string):
        if generate_method == 'normalvariate':
            value = round(random.normalvariate(mean, dispersion), 2)
        elif generate_method == 'lognormvariate':
            value = round(random.lognormvariate(mean, dispersion), 2)
        else:
            value = round(random.random(), 2)

        if value < 0: return 0
        if value > 1: return 1
        return value


    matrix_size = a_size + b_size + c_size
    compatibility_matrix = [[0 for x in range(matrix_size)] for y in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(i, matrix_size):
            compatibility_matrix[i][j] = generate_value(generate_method)

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
