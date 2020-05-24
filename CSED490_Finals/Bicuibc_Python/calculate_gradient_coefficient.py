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
original_image = [255, 240, 222, 252, 202, 235, 234, 230, 222, 269, 215]
""" 
    !!! --- End of variable definition --- !!!
--------------------------------------------------
"""

""" 
--------------------------------------------------
    !!! --- Define FUNCTIONS here --- !!!
"""


def gradient_coefficient(i, o_img):
    input_matrix = []
    coefficient_matrix = []

    " Generate input matrix & coefficient matrix"
    for j in range(i - 1, i + 3, 1):
        print(j)
        if j < 0 or j > len(o_img) - 1:
            input_matrix.append(o_img[len(o_img) - 1])
            coefficient_row = [(j**3), (j**2), j, 1]

        else:
            input_matrix.append(o_img[j])
            coefficient_row = [(j**3), (j**2), j, 1]

        coefficient_matrix.append(coefficient_row)

    #coefficient_matrix = np.array(coefficient_matrix)
    #input_matrix = np.array(input_matrix)

    return coefficient_matrix, input_matrix, np.linalg.solve(coefficient_matrix, input_matrix)


""" 
        !!! --- End of functions --- !!!
--------------------------------------------------
"""

""" 
--------------------------------------------------
    !!! --- Define Operations here --- !!!
"""
x = 9
scale = 2
a,b,grad = gradient_coefficient(x, original_image)
new_pixel = grad[0] * ((x + (1/scale)) ** 3) + grad[1] * ((x + (1/scale)) ** 2) + grad[2] * (x + (1/scale)) + grad[3]
print("{}\n".format(a))
print('{}\n'.format(b))
print('{}\n'.format(grad))
print('{}\n'.format(new_pixel))
""" 
        !!! --- End of Operations --- !!!
--------------------------------------------------
"""
