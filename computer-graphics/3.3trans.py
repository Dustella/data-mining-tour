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

    def transform_scale(self, scale):
        return Point_3d(self.x*scale, self.y*scale, self.z*scale)

    def transform_move(self, x, y, z):
        return Point_3d(self.x+x, self.y+y, self.z+z)

    def transform_rotate(self, yaw, pitch, roll):
        R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll),
                       np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                      [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll),
                       np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                      [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
        xx, yy, zz = np.dot(R, np.array([self.x, self.y, self.z]))
        return Point_3d(xx, yy, zz)

    def transform_project(self, a, b, c):
        xx = self.x * np.cos(a) + self.y * np.sin(a)
        yy = self.x * np.cos(b) + self.z * np.sin(b)
        zz = self.y * np.cos(c) + self.z * np.sin(c)
        return Point_3d(xx, yy, zz)

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

    def clear(self):
        self.image = np.array([[0 for i in range(1000)]
                               for i in range(1000)], copy=False)/255

    def draw_line(self, p1: Point_2d, p2: Point_2d):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        if x1 == x2:
            # 垂直线
            for i in range(min(y1, y2), max(y1, y2)):
                self.image[i, x1] = 1
        else:
            # 非垂直线
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            # 画出直线
            for i in range(min(x1, x2), max(x1, x2)):
                self.image[int(k * i + b), i] = 1


class Cube:
    def __init__(self, p1: Point_3d, p8: Point_3d) -> None:
        p2 = Point_3d(p1.x, p1.y, p8.z)
        p3 = Point_3d(p1.x, p8.y, p1.z)
        p4 = Point_3d(p1.x, p8.y, p8.z)
        p5 = Point_3d(p8.x, p1.y, p1.z)
        p6 = Point_3d(p8.x, p1.y, p8.z)
        p7 = Point_3d(p8.x, p8.y, p1.z)

        self.pls = p1, p2, p3, p4, p5, p6, p7, p8

    def rotate(self):
        return Cube(*map(lambda x: x.transform_rotate(3.14/6, 3.14/6, 3.14/6), [self.pls[0], self.pls[7]]))

    def move(self, x, y, z):
        return Cube(*map(lambda a: a.transform_move(x, y, z), [self.pls[0], self.pls[7]]))

    def scale(self, scale):
        return Cube(*map(lambda a: a.transform_scale(scale), [self.pls[0], self.pls[7]]))

    def project(self):
        return Cube(*map(lambda a: a.transform_project(3.14/4, 3.14/4, 3.14/4), [self.pls[0], self.pls[7]]))

    def draw(self, r: Render):
        # map all points to 2d
        p2d = list(map(lambda x: x.project_to_2d(3.14/4, 3.14/4), self.pls))
        # draw lines
        for i in self.pls:
            print(i)
        for i in p2d:
            print(i)
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
    r.clear()
    cube.draw(r)
    cube.scale(0.5).draw(r)
    cv2.imshow('image_scale', r.image)
    cv2.waitKey(0)
    r.clear()
    cube.draw(r)
    cube.rotate().draw(r)
    cv2.imshow('image_rotate', r.image)
    cv2.waitKey(0)
    r.clear()
    cube.draw(r)
    cube.project().draw(r)
    cv2.imshow('image_project', r.image)
    cv2.waitKey(0)
    r.clear()
