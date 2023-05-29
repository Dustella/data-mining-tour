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


# 显示绘制好的图像
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
