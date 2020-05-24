"""
--------------------------------------------------
!!! --- Import required LIBRARIES here --- !!!
"""
import numpy as np

""" 
    !!! --- End of libraries import --- !!!
--------------------------------------------------
"""

""" 
--------------------------------------------------
    !!! --- Define VARIABLES here --- !!!
"""
coefficient_matrix = [[0, 0, 0, 1],
                               [1, 1, 1, 1],
                               [8, 4, 2, 1],
                               [27, 9, 3, 1]]
input_matrix = [90, 160, 244, 32]
""" 
    !!! --- End of variable definition --- !!!
--------------------------------------------------
"""

""" 
--------------------------------------------------
    !!! --- Define FUNCTIONS here --- !!!
"""


def linear_eqn_solver(A, B):
    return np.linalg.solve(A, B)


""" 
        !!! --- End of functions --- !!!
--------------------------------------------------
"""

""" 
--------------------------------------------------
    !!! --- Define Operations here --- !!!
"""
arr = linear_eqn_solver(coefficient_matrix, input_matrix)
a = arr[0]
b = arr[1]
c = arr[2]
d = arr[3]
print("[a, b, c, d] = [{}, {}, {}, {}]".format(a, b, c, d))
""" 
        !!! --- End of Operations --- !!!
--------------------------------------------------
"""
