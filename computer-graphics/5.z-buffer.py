import cv2
import numpy as np

# 创建灰度图像
height = 480
width = 640
img = np.zeros((height, width), dtype=np.uint8)

# 创建窗口并显示图像

# 绘制长方体
l, w, h = 100, 80, 60
cx, cy, cz = width // 2, height // 2, 1000  # 长方体中心点的位置

# 定义顶点坐标
vertices = np.array([(cx-l/2, cy-w/2, cz-h/2),
                     (cx+l/2, cy-w/2, cz-h/2),
                     (cx+l/2, cy+w/2, cz-h/2),
                     (cx-l/2, cy+w/2, cz-h/2),
                     (cx-l/2, cy-w/2, cz+h/2),
                     (cx+l/2, cy-w/2, cz+h/2),
                     (cx+l/2, cy+w/2, cz+h/2),
                     (cx-l/2, cy+w/2, cz+h/2)])

# 定义边的连接方式
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

# 定义每条边的深度值
depth = np.array([cz-h/2, cz-h/2, cz-h/2, cz-h/2, cz+h/2,
                 cz+h/2, cz+h/2, cz+h/2, cz-h/2, cz-h/2, cz-h/2, cz-h/2])

# 对每条边按照深度值进行排序
sorted_idx = np.argsort(depth)

# 创建z-buffer
z_buffer = np.full((height, width), np.inf)

# 依次绘制每条边，并对边进行消隐
for i in sorted_idx:
    x1, y1, z1 = vertices[edges[i][0]]
    x2, y2, z2 = vertices[edges[i][1]]

    # 将3D点转换为2D点
    u1, v1 = int(x1/z1*cz + cx), int(y1/z1*cz + cy)
    u2, v2 = int(x2/z2*cz + cx), int(y2/z2*cz + cy)

    # 使用传统的扫描线算法画线
    if u1 == u2:
        continue
    if u1 > u2:
        u1, v1, u2, v2 = u2, v2, u1, v1

    # 计算斜率
    k = (v2 - v1) / (u2 - u1)

    # 画线
    for u in range(u1, u2):
        v = int(v1 + k * (u - u1))
        if z_buffer[v, u] > 1/z1:
            z_buffer[v, u] = 1/z1
            img[v, u] = 255


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
    def transform_project(self, a, b, c):
        xx = self.x * np.cos(a) + self.y * np.sin(a)
        yy = self.x * np.cos(b) + self.z * np.sin(b)
        zz = self.y * np.cos(c) + self.z * np.sin(c)
        return Point_3d(xx, yy, zz)
    
    def project_to_2d(self, a, b):
        return Point_2d(self.x * np.cos(a) + self.y * np.sin(a),
                        self.x * np.cos(b) + self.z * np.sin(b))

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'





class Cube:
    def __init__(self, p1: Point_3d, p8: Point_3d) -> None:
        p2 = Point_3d(p1.x, p1.y, p8.z)
        p3 = Point_3d(p1.x, p8.y, p1.z)
        p4 = Point_3d(p1.x, p8.y, p8.z)
        p5 = Point_3d(p8.x, p1.y, p1.z)
        p6 = Point_3d(p8.x, p1.y, p8.z)
        p7 = Point_3d(p8.x, p8.y, p1.z)

        self.pls = p1, p2, p3, p4, p5, p6, p7, p8

# return list of FaceBuffer
    def get_buffer(self)-> list :
        # map all points to 2d
        p2d = list(map(lambda x: x.transform_project(3.14/4, 3.14/4,3.14/4), self.pls))
        # draw lines
        for i in self.pls:
            print(i)
        for i in p2d:
            print(i)
        # init face buffer
        face_buffer = []
        face_buffer.append(FaceBuffer(p2d[0], p2d[1], p2d[2], p2d[3], 1/9))
        face_buffer.append(FaceBuffer(p2d[0], p2d[1], p2d[4], p2d[5], 2/9))
        face_buffer.append(FaceBuffer(p2d[0], p2d[2], p2d[4], p2d[6], 3/9))
        face_buffer.append(FaceBuffer(p2d[1], p2d[3], p2d[5], p2d[7], 4/9))
        face_buffer.append(FaceBuffer(p2d[2], p2d[3], p2d[6], p2d[7], 5/9))
        face_buffer.append(FaceBuffer(p2d[4], p2d[5], p2d[6], p2d[7], 6/9))
        return face_buffer

class FaceBuffer:
    z:int
    
    def __init__(self, p1:Point_3d,p2:Point_3d, p3: Point_3d, p4:Point_3d,color=1 ) -> None:
        # z is average of 4 points
        self.z = (p1.z+p2.z+p3.z+p4.z)/4
        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4
        self.buffer = np.array([[0 for i in range(1000)] for i in range(1000)], copy=False)/255
        self.color = color
        self.render()
        

    def render(self):
        # draw lines, then fill
        self.draw_line(self.p1, self.p2)
        self.draw_line(self.p2, self.p4)
        self.draw_line(self.p4, self.p3)
        self.draw_line(self.p1, self.p3)
        self.fill(self.color)
    
    def draw_line(self, p1:Point_2d, p2:Point_2d):
        # 使用扫描线算法
        x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
        # sort in case of x1>x2
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if x1 == x2:
            for i in range(y1, y2):
                self.buffer[x1][i] = 1
        else:
            k = (y2-y1)/(x2-x1)
            for i in range(x1, x2):
                self.buffer[i][int(k*(i-x1)+y1)] = 1

    def fill(self,color=1):
        # 使用扫描线算法
        for x in range(1000):
            start_y, end_y = 0, 0
            for y in range(1000):
                # search for not 0 point
                if self.buffer[x][y] != 0:
                    if start_y == 0:
                        start_y = y
                    else:
                        end_y = y
            if start_y != 0 and end_y != 0:
                for i in range(start_y, end_y):
                    self.buffer[x][i] = color
    
    # debug: show buffer
    def show(self):
        cv2.imshow('image', self.buffer)
        cv2.waitKey(0)

    
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

    def draw_cube(self,c:Cube):
        buffer_ls = c.get_buffer()
        for face_buffer in buffer_ls:
            face_buffer.show()
            for x in range(1000):
                for y in range(1000):
                    if face_buffer.buffer[x][y] != 0:
                        if face_buffer.z < self.z_buffer[x][y]:
                            self.image[x][y] = face_buffer.buffer[x][y]
                            self.z_buffer[x][y] = face_buffer.z



if __name__ == "__main__":
    r = Render()
    # # 画一个立方体
    cube = Cube(Point_3d(100, 120, 120), Point_3d(400, 400, 400))
    r.draw_cube(cube)
    cv2.imshow('image', r.image)
    cv2.waitKey(0)


    # p1 = Point_3d(100, 100, 100)  
    # p2 = Point_3d(150, 50, 200)
    # p3 = Point_3d(200, 100, 100)
    # p4 = Point_3d(150, 150, 200)

    # fb = FaceBuffer(p1, p2, p3, p4,0.5)
    # fb.render()
    # fb.show()
