import cv2 as cv
import numpy as np

image = np.array([[0 for i in range(1000)]
                 for i in range(1000)], copy=False)/255

image_list = []

# 使用扫描线法绘制直线


def draw_line(image, x1, y1, x2, y2):
    # make floats into integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 == x2:
        # 垂直线
        for i in range(min(y1, y2), max(y1, y2)):
            image[i, x1] = 1
            image_clone = image.copy()
            image_list.append(image_clone)
    else:
        # 非垂直线
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # 画出直线
        for i in range(min(x1, x2), max(x1, x2)):
            image[int(k * i + b), i] = 1
            image_list.append(image.copy())

# 绘制三角形


def draw_triangle(image, matrix):
    [[x1, y1, _], [x2, y2, _], [x3, y3, _]] = matrix
    draw_line(image, x1, y1, x2, y2)
    draw_line(image, x2, y2, x3, y3)
    draw_line(image, x3, y3, x1, y1)

# do transform to the vector to rotate it with the angle


def rotate_transform(vector, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
    return np.dot(matrix, vector)

# do transform to the vector to scale it with the scale


def scale_transform(vector, scale):
    matrix = np.array([[scale, 0, 0],
                       [0, scale, 0],
                       [0, 0, 1]])
    return np.dot(matrix, vector)

#  do transform to the vector to translate it with the offset


def translate_transform(vector, offset):
    matrix = np.array([[1, 0, offset[0]],
                       [0, 1, offset[1]],
                       [0, 0, 1]])
    return np.dot(matrix, vector)


origin_triangle = np.array([[500, 100, 1], [600, 400, 1], [400, 400, 1]])

# rotate the triangle with
rotated_triangle = map(lambda vec: rotate_transform(
    vec, np.pi/4), origin_triangle)

# scale the triangle with 1.4
scaled_triangle = map(lambda vec: scale_transform(vec, 1.4), origin_triangle)

# translate the triangle with (100, 100)
translated_triangle = map(
    lambda vec: translate_transform(vec, (100, 100)), origin_triangle)

# draw the triangle
draw_triangle(image, origin_triangle)
draw_triangle(image, rotated_triangle)
draw_triangle(image, scaled_triangle)
draw_triangle(image, translated_triangle)


# 使用cv2.imshow()显示图像
for img in image_list:
    cv.imshow('image', img)
    cv.waitKey(2)

cv.waitKey(0)
cv.destroyAllWindows()
