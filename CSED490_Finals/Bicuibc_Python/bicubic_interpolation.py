""" !!! --- Import required LIBRARIES here --- !!! """

import numpy as np

""" !!! --- Define VARIABLES here --- !!! """

original_image = [240, 200, 0, 165, 177, 80, 99, 254, 2, 32, 50, 103, 91, 193, 128, 55]

""" !!! --- Define FUNCTIONS here --- !!! """


def gradient_coefficient(i, o_img):
    input_matrix = []
    coefficient_matrix = []

    " Generate input matrix & coefficient matrix"
    for j in range(i - 1, i + 3, 1):
        if j < 0 or j > len(o_img) - 1:
            coefficient_row = [(j ** 3), (j ** 2), j, 1]
            input_matrix.append(o_img[len(o_img) - 1])

        else:
            coefficient_row = [(j ** 3), (j ** 2), j, 1]
            input_matrix.append(o_img[j])

        coefficient_matrix.append(coefficient_row)

    input_matrix = np.array(input_matrix)
    coefficient_matrix = np.array(coefficient_matrix)

    return np.linalg.solve(coefficient_matrix, input_matrix)


def image_rescale(scale, old_img):
    new_img = []
    for i in range(len(old_img)):
        " Append original pixel into new image "
        new_img.append(old_img[i])

        " Generate gradients of cubic eqn "
        grad = gradient_coefficient(i, old_img)

        " Generate new pixels & append to new image "
        for j in range(1, scale, 1):
            xn = i + j * (1 / scale)
            new_pixel = grad[0] * (xn ** 3) + grad[1] * (xn ** 2) + grad[2] * xn + grad[3]

            if new_pixel < 0:
                new_pixel = 0

            elif new_pixel > 255:
                new_pixel = 255

            new_pixel = int(new_pixel)
            new_img.append(new_pixel)

    return old_img, new_img


""" !!! --- Define FUNCTIONS here --- !!! """

scale = 2
image = image_rescale(scale, original_image)
print("Original image: {}. \nResized image: {}.".format(image[0], image[1]))
