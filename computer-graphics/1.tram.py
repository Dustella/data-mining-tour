import cv2 as cv
import numpy as np

image = np.array([[0 for i in range(1000)]
                 for i in range(1000)], copy=False)/255

image_list = []

# 使用扫描线法绘制直线


def draw_line(image, x1, y1, x2, y2):
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


# 画圆


def draw_circle(image, x, y, r):
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            if (i-x)**2 + (j-y)**2 <= r**2 and (i-x)**2 + (j-y)**2 >= (r-1)**2:
                image[j, i] = 1
                image_list.append(image.copy())

# 画弧函数


def draw_arc(image, x, y, r, start, end):
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            if (i-x)**2 + (j-y)**2 <= r**2 and (i-x)**2 + (j-y)**2 >= (r-1)**2:
                if np.arctan2(j-y, i-x) >= start and np.arctan2(j-y, i-x) <= end:
                    image[j, i] = 1
                    image_list.append(image.copy())


# 画出一个五角星
draw_line(image, 500, 100, 600, 400)


# 画一个笑脸
draw_circle(image, 300, 300, 100)
draw_circle(image, 250, 270, 10)
draw_circle(image, 350, 270, 10)
draw_arc(image, 300, 300, 50, 0, np.pi)


# 使用cv2.imshow()显示图像
for img in image_list:
    cv.imshow('image', img)
    cv.waitKey(2)

cv.destroyAllWindows()
