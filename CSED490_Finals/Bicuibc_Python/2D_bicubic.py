""" !!! --- Import required LIBRARIES here --- !!! """
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time
""" !!! --- Define FUNCTIONS here --- !!! """


def random_square_image(width):
    random.seed()
    img = np.array([])

    for y in range(width):
        img_row = np.array([])

        for x in range(width):
            random_pixel = random.randint(0, 255)
            img_row = np.append(img_row, random_pixel)

        if y == 0:
            img = np.array(img_row)

        else:
            img = np.vstack((img, img_row))

    return img.astype(int)


def generate_weights():
    a = [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [-3, 3, -2, -1],
         [2, -2, 1, 1]]

    b = [[1, 0, -3, 2],
         [0, 0, 3, -2],
         [0, 1, -2, 1],
         [0, 0, -1, 1]]

    return a, b


def input_matrix(old_image, x, y, x_dim, y_dim):
    in_m = []

    for yi in range(-1, 3, 1):
        yj = y + yi
        in_row = []

        for xi in range(-1, 3, 1):
            xj = x + xi
            if xj < 0 or yj < 0 or xj > x_dim - 1 or yj > y_dim - 1:
                in_row.append(0)

            else:
                in_row.append(old_image[y + yi][x + xi])

        in_m.append(in_row)
    return in_m


def generate_gradient(input_array, weights):
    gradient_weight = np.dot(np.dot(weights[0], input_array), weights[1])

    return gradient_weight


def multiply_pixel(scale, gradient_matrix):
    pixel_array = []

    for y in np.arange(1/scale, 1.1, 1/scale):
        y_a = [1, y, y ** 2, y ** 3]
        y_array = np.transpose(y_a)
        pixel_row = []

        for x in np.arange(1/scale, 1.1, 1/scale):
            x_array = [1, x, x ** 2, x ** 3]

            new_pixel = np.dot(np.dot(x_array, gradient_matrix), y_array)

            pixel_row.append(new_pixel)

        pixel_array.append(pixel_row)

    return pixel_array


def bicubic_2D(old_image, scale):
    y_dim = len(old_image)
    x_dim = len(old_image[0])
    weights = generate_weights()
    new_image = np.array([])

    for y in range(y_dim):
        pixel_row = np.array([])
        for x in range(x_dim):
            input_array = input_matrix(old_image, x, y, x_dim, y_dim)
            gradient_weight = generate_gradient(input_array, weights)
            new_pixel = multiply_pixel(scale, gradient_weight)

            if x == 0:
                pixel_row = new_pixel

            else:
                pixel_row = np.append(pixel_row, new_pixel, axis=1)

        if y == 0:
            new_image = pixel_row

        else:
            new_image = np.vstack((new_image, pixel_row))

    return new_image.astype(int)


"""  !!! --- Define Operations here --- !!! """
start_time = time.time() * 1000

w = 512
o_img = random_square_image(w)
print(o_img)

next_time = time.time() * 1000
img_gen_latency = next_time - start_time
print("\nImage took {}ms to create!\n".format(img_gen_latency))

s = 4
n_img = bicubic_2D(o_img, s)
print(n_img)

next_time = time.time() * 1000
total_latency = next_time - start_time
img_resize_latency = total_latency - img_gen_latency
print("\nImage resize took {}ms to compute!".format(img_resize_latency))
print("Total computation time is {}ms!".format(total_latency))

'''cv2.imwrite('o_img.jpg', o_img)
cv2.imwrite('n_img.jpg', n_img)

oo_img = cv2.imread("o_img.jpg", 0)
nn_img = cv2.imread("n_img.jpg", 0)

cv2.imshow("Original Image", oo_img)
cv2.imshow("Resized Image", nn_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(o_img, cmap='gray', vmin=0, vmax=255)
ax1.set_title("Original Image")
ax2.imshow(n_img, cmap='gray', vmin=0, vmax=255)
ax2.set_title("Upsized Image")
plt.show()'''
