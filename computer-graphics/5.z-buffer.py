import cv2
import numpy as np


class Point_2d:
    x: int
    y: int

    def __init__(self, x, y) -> None:
        self.x, self.y = int(x), int(y)

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'


class Point_3d:
    x: int
    y: int
    z: int

    def __init__(self, x, y, z) -> None:
        self.x, self.y, self.z = int(x), int(y), int(z)
        pass

    def value(self):
        return [self.x, self.y, self.z]

    # 投影到二维平面

    def project_to_2d(self, a, b):
        return Point_2d(self.x * np.cos(a) + self.y * np.sin(a),
                        self.x * np.cos(b) + self.z * np.sin(b))

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'


class Render:
    def __init__(self) -> None:
        self.image = np.array([[0 for i in range(1000)]
                               for i in range(1000)], copy=False)/255
        self.z_buffer = np.array([[float('inf') for i in range(1000)]
                                 for i in range(1000)], copy=False)/255

    def clear(self):
        self.image = np.array([[0 for i in range(1000)]
                               for i in range(1000)], copy=False)/255
        self.z_buffer = np.array([[float('inf') for i in range(1000)]
                                  for i in range(1000)], copy=False)/255


class Buffer:
    """Buffer for Z-buffer and other buffers"""

    def __init__(self, width=1000, height=1000):
        self.width = width
        self.height = height
        self.buffer = np.array([[0 for i in range(width)]
                                for i in range(height)], copy=False)/255
        self.z_buffer = np.array(
            [[float('inf') for i in range(width)] for i in range(height)], copy=False)/255

    def _draw_line(self, p1: Point_2d, p2: Point_2d):
        # 扫描法画线
        # 如果是垂直线
        if p1.x == p2.x:
            for i in range(min(p1.y, p2.y), max(p1.y, p2.y)):
                self.buffer[i, p1.x] = 1
        else:
            # 非垂直线
            k = (p2.y - p1.y) / (p2.x - p1.x)
            b = p1.y - k * p1.x
            # 画出直线
            for i in range(min(p1.x, p2.x), max(p1.x, p2.x)):
                self.buffer[int(k * i + b), i] = 1

    def fill(self, p1: Point_2d, p2: Point_2d, p3: Point_2d):
        # 画出三条边
        self._draw_line(p1, p2)
        self._draw_line(p1, p3)
        self._draw_line(p2, p3)
        # 用扫描法填充
        # 找到最高点和最低点
        p1, p2, p3 = sorted([p1, p2, p3], key=lambda x: x.y)
        # 扫描
        for i in range(p1.y, p3.y):
            # 找到两条边的交点
            x1 = (i-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x
            x2 = (i-p1.y)*(p3.x-p1.x)/(p3.y-p1.y)+p1.x
            # 画线
            self._draw_line(Point_2d(x1, i), Point_2d(x2, i))

        pass


class Cube:
    def __init__(self, p1: Point_3d, p8: Point_3d) -> None:
        p2 = Point_3d(p1.x, p1.y, p8.z)
        p3 = Point_3d(p1.x, p8.y, p1.z)
        p4 = Point_3d(p1.x, p8.y, p8.z)
        p5 = Point_3d(p8.x, p1.y, p1.z)
        p6 = Point_3d(p8.x, p1.y, p8.z)
        p7 = Point_3d(p8.x, p8.y, p1.z)

        self.pls = p1, p2, p3, p4, p5, p6, p7, p8

    def draw(self, r: Render):
        # map all points to 2d
        p2d = list(map(lambda x: x.project_to_2d(3.14/4, 3.14/4), self.pls))
        # draw lines
        for i in self.pls:
            print(i)
        for i in p2d:
            print(i)

        # 计算深度值
        for i in range(8):
            r.z_buffer[p2d[i].y, p2d[i].x] = self.pls[i].z

        r.draw_line(p2d[0], p2d[1])
        r.draw_line(p2d[0], p2d[2])
        r.draw_line(p2d[0], p2d[4])
        r.draw_line(p2d[1], p2d[3])

        r.draw_line(p2d[1], p2d[5])
        r.draw_line(p2d[2], p2d[3])
        r.draw_line(p2d[2], p2d[6])
        r.draw_line(p2d[3], p2d[7])

        r.draw_line(p2d[4], p2d[5])
        r.draw_line(p2d[4], p2d[6])
        r.draw_line(p2d[5], p2d[7])
        r.draw_line(p2d[6], p2d[7])


if __name__ == "__main__":
    r = Render()
    # 画一个立方体
    cube = Cube(Point_3d(100, 120, 120), Point_3d(400, 400, 400))
    cube.draw(r)
    cube.move(50, 50, 10).draw(r)
    cv2.imshow('image_move', r.image)
    cv2.waitKey(0)

    cv2.imshow('image_project', r.image)
    cv2.waitKey(0)
    r.clear()
