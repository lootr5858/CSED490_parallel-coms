import random
import numpy as np


def generate_matrix(mini=0, maxi=10, row=1, col=1):
    random.seed()
    matrix = []
    matrix_b = []

    for y in range(row):
        sub_row = []
        for x in range(col):
            sub_row.append(random.randint(mini, maxi))

        matrix.append(sub_row)

    return matrix


def matrix_multiplication(matrix_a, matrix_b):
    matrix_c = []
    """ Extract dimension of matrices """
    a_row = len(matrix_a)
    a_col = len(matrix_a[0])

    b_row = len(matrix_b)
    b_col = len(matrix_b[0])

    """ Check if matrix can be multiplied """
    can_multiply = None
    if a_col == b_row:
        can_multiply = True

    else:
        can_multiply = False

    """ Matrix multiplication operations """
    if can_multiply:
        for i in range(a_row):
            sub_c = []
            for j in range(b_col):
                partial_sum = 0
                for k in range(a_col):
                    partial_sum += matrix_a[i][k] * matrix_b[k][j]
                sub_c.append(partial_sum)
            matrix_c.append(sub_c)
        return matrix_c

    else:
        return "Wrong dimensions!!!"


a = generate_matrix(row=2, col=4)
b = generate_matrix(row=3, col=3)
print(np.array(a))
print(np.array(b))

c = matrix_multiplication(a, b)
print(np.array(c))
