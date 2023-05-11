import cv2 as cv
import numpy as np

image = np.array([[0 for i in range(1000)]
                 for i in range(1000)], copy=False)/255


def draw_line(x1, y1, x2, y2):
    if x1 == x2:
        # 垂直线
        for i in range(min(y1, y2), max(y1, y2)):
            image[i, x1] = 1
    else:
        # 非垂直线
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # 画出直线
        for i in range(min(x1, x2), max(x1, x2)):
            image[int(k * i + b), i] = 1


def draw_bizear(x1, y1, x2, y2, x3, y3, x4, y4):
    # 我们使用1000个点来画贝塞尔曲线
    for t in range(0, 1000):
        # 画出控制点
        t = t/1000
        x = (1-t)**3*x1+3*t*(1-t)**2*x2+3*t**2*(1-t)*x3+t**3*x4
        y = (1-t)**3*y1+3*t*(1-t)**2*y2+3*t**2*(1-t)*y3+t**3*y4
        image[int(y), int(x)] = 1


draw_bizear(100, 100, 200, 200, 300, 100, 400, 200)
draw_line(100, 100, 200, 200)
draw_line(300, 100, 400, 200)

draw_bizear(400, 200, 500, 300, 600, 200, 700, 500)
draw_line(400, 200, 500, 300)
draw_line(600, 200, 700, 500)

cv.imshow('image', image)
cv.waitKey(0)

cv.destroyAllWindows()
